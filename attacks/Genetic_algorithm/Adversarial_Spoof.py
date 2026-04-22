#!/usr/bin/env python3
"""
Adversarial Spoofing Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on RPM/Spoofing attack scenarios by modifying ECU-controlled dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

Key characteristics of Spoofing attacks:
- Uses ECU control information to identify modifiable dummy rows
- Only modifies rows where ECU control value = 1 (transmitter controlled)
- Similar to DoS attacks but with ECU-specific constraints
- Preserves RPM attack payload in non-dummy rows

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_spoofing_attack.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from attacks.attack_handler.base import GeneticAttack

class AdversarialSpoofAttack(GeneticAttack):
    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial Spoofing attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """

        super().__init__(model_path, file_path, population_size, max_generations, mutation_rate)
        self.original_dummy_rows = []
        self.ecu_control = self.data['ecu_control']  # ECU control values for each frame

    def find_dummy_rows(self, frame_idx):
        """
        Find and return indices of ECU-controlled dummy rows for a specific frame.
        
        For spoofing attacks, dummy rows are identified by ECU control values:
        - ECU control value = 0: Receiver controlled (cannot modify)
        - ECU control value = 1: Transmitter controlled (can modify)
        
        This is specific to RPM/spoofing attacks where ECU control information
        determines which parts of the CAN frame can be safely modified without
        destroying the attack payload.
        
        Args:
            frame_idx: Index of the frame in the dataset
            
        Returns:
            List of row indices where ECU control value = 1 (modifiable rows)
        """
        if frame_idx >= len(self.ecu_control):
            return []  # Return empty list if index is out of bounds
            
        return [j for j in range(29) if self.ecu_control[frame_idx][j] == 1]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original ECU-controlled dummy rows.
        
        Spoofing attack mutation strategy:
        - Only modifies rows that were originally identified as ECU-controlled (value=1)
        - For each dummy row, applies mutation with probability = mutation_rate
        - Sets exactly one random bit to 1 in the selected row
        - Unlike fuzzy attacks, can set any bit position (0-28) including priority bits
        
        This maintains the constraint that only ECU-transmitter controlled portions
        of the frame can be modified, preserving the RPM attack characteristics.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        for i in self.original_dummy_rows:
            if random.random() < self.mutation_rate:
                bit_to_flip = random.randint(0, 28)
                mutated[i, bit_to_flip, 0] = 1
                
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Spoofing attack crossover strategy:
        - Start with parent1 as the base (child inherits most characteristics)
        - For each original ECU-controlled dummy row, randomly choose to inherit from parent2
        - Only swap rows that were originally ECU-controlled (preserves attack payload)
        - 50% chance per row to inherit from parent2
        
        This allows combining successful mutations from different parents
        while maintaining the integrity of the original RPM attack structure
        and ECU control constraints.
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining characteristics from both parents
        """
        child = parent1.copy()
        
        for i in self.original_dummy_rows:
            if random.random() < 0.5:  # 50% chance to inherit from parent2
                child[i] = parent2[i]
                
        return child

    def calculate_confidence(self, frame):
        """
        Calculate the IDS confidence score for classifying a frame as an attack.
        
        The IDS model outputs probabilities for [normal, attack] classes.
        We return the attack confidence (index 1) since our goal is to
        minimize this score below 0.5 to achieve misclassification.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Float between 0-1 representing attack confidence score
        """
        frame_batch = np.expand_dims(frame, 0)
        
        prediction = self.model.predict(frame_batch, verbose=0)
        
        return prediction[0][1]

    def generate_adversarial_attack(self, dummy_row_threshold=1, max_frames=100):
        """
        Main genetic algorithm to generate adversarial Spoofing attacks.
        
        Process:
        1. Create balanced dataset (70% attack, 30% benign)
        2. Add benign frames directly (no modification needed)
        3. For each attack frame with sufficient ECU-controlled dummy rows:
           - Identify ECU-controlled dummy rows using ECU control data
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum ECU-controlled rows required to process a frame
            max_frames: Maximum total frames to include in final dataset
            
        Returns:
            tuple: (final_test, y_test, orig_frame, generations_needed)
                - final_test: Adversarial frames ready for evaluation
                - y_test: Corresponding labels for the frames
                - orig_frame: Original frames before adversarial modification
                - generations_needed: List of generations required per attack frame
        """
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count
        
        print(f"Using {attack_count} attack frames and {benign_count} benign frames")
        
        benign_indices = np.where(self.y_test == 0)[0][:benign_count]
        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")
        
        attack_indices = np.where(self.y_test == 1)[0]
        if len(attack_indices) < attack_count:
            attack_count = len(attack_indices)
            print(f"Warning: Only {attack_count} attack frames available")
        
        orig_frame = []          # Original frames for comparison
        final_test = []          # Adversarial/benign frames for evaluation
        generations_needed = []  # Track genetic algorithm performance
        
        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(self.x_test[i])
        
        print("Starting genetic algorithm for adversarial attack generation...")
        
        attack_frames_processed = 0
        for i in attack_indices:
            if attack_frames_processed >= attack_count:
                break
                
            dummy_rows = self.find_dummy_rows(i)
            
            if len(dummy_rows) < dummy_row_threshold:
                continue  # Skip this frame
                final_test.append(self.x_test[i])
                orig_frame.append(self.x_test[i])
                attack_frames_processed += 1
            else:
                self.original_dummy_rows = dummy_rows.copy()
                frame_copy = self.x_test[i].copy()
                
                mut_rate = self.mutation_rate  # Save current mutation rate
                self.mutation_rate = 1         # Force mutation for population diversity
                
                population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
                
                self.mutation_rate = mut_rate
                
                success_generation = -1  # Track when successful attack was found
                
                for generation in range(self.max_generations):
                    print(f"Attack frame {attack_frames_processed+1}/{attack_count}, Generation {generation+1}/{self.max_generations}")

                    # Batch predict all individuals at once
                    scores = self.calculate_confidence_batch(population)
                    scores = np.nan_to_num(scores, nan=1.0)

                    # Check for successful evasion
                    success_idx = np.where(scores < 0.5)[0]
                    if len(success_idx) > 0:
                        winner = success_idx[0]
                        print(f"Successful attack found in generation {generation+1}")
                        final_test.append(population[winner])
                        orig_frame.append(self.x_test[i])
                        success_generation = generation + 1
                        break

                    inv_scores = 1.0 - scores

                    total = inv_scores.sum()
                    num_positive = np.count_nonzero(inv_scores)

                    if total <= 1e-12 or num_positive < 2:
                        selection_probs = np.ones_like(inv_scores) / len(inv_scores)
                    else:
                        selection_probs = inv_scores / total

                    indices = np.arange(len(population))
                    new_pop = []

                    best_idx = np.argmin(scores)
                    new_pop.append(population[best_idx])

                    while len(new_pop) < self.population_size:
                        p1_idx, p2_idx = np.random.choice(
                            indices, size=2, p=selection_probs, replace=False
                        )

                        child = self.crossover(
                            population[p1_idx],
                            population[p2_idx]
                        )

                        child = self.mutate(child)
                        new_pop.append(child)

                    population = new_pop

                if success_generation == -1:
                    best_idx = np.argmin(scores)
                    print(f"Best score achieved: {scores[best_idx]:.4f}")
                    final_test.append(population[best_idx])
                    orig_frame.append(self.x_test[i])
                    success_generation = self.max_generations
                
                generations_needed.append(success_generation)
                attack_frames_processed += 1
        
        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def run_dummy_row_experiment(self,model_path, max_frames=20):
        """
        Experimental function to test the effect of ECU-controlled dummy row threshold on attack success.
        
        Tests different minimum ECU-controlled row requirements and measures performance.
        Higher thresholds mean:
        - More ECU-controlled modification space available (easier attacks)
        - Fewer eligible frames (reduced dataset size)
        
        Args:
            model_path: Path to the trained IDS model
            max_frames: Number of frames to test per threshold
            
        Returns:
            Dictionary mapping dummy row thresholds to average generations needed
        """
        dummy_row_thresholds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        results = {}
        
        attack_obj = AdversarialSpoofAttack(model_path=model_path)
        
        for threshold in dummy_row_thresholds:
            print(f"\n--- Testing dummy row threshold: {threshold} ---")
            
            suitable_frames = []
            for i in range(min(len(attack_obj.x_test), len(attack_obj.ecu_control))):
                if attack_obj.y_test[i] == 1:  # Only consider attack frames
                    dummy_rows = attack_obj.find_dummy_rows(i)
                    if len(dummy_rows) >= threshold:  # Frame has enough ECU-controlled rows
                        suitable_frames.append(i)
            
            print(f"Found {len(suitable_frames)} frames with dummy rows >= {threshold}")
            
            if len(suitable_frames) == 0:
                results[threshold] = 0
                print(f"Dummy row threshold {threshold}: No suitable frames found")
                continue
                
            suitable_frames = suitable_frames[:max_frames]
            
            attack = AdversarialSpoofAttack(
                model_path=model_path,
                population_size=100,
                max_generations=75,
                mutation_rate=0.3  # Fixed mutation rate for fair comparison
            )
            
            generations_list = []
            for idx, frame_idx in enumerate(suitable_frames):
                print(f"Processing frame {idx+1}/{len(suitable_frames)}")
                
                frame = attack.x_test[frame_idx].copy()
                dummy_rows = attack.find_dummy_rows(frame_idx)
                attack.original_dummy_rows = dummy_rows.copy()
                
                population = [attack.mutate(frame.copy()) for _ in range(attack.population_size)]
                
                success_generation = -1
                for generation in range(attack.max_generations):
                    # Batch predict all individuals at once
                    scores = attack.calculate_confidence_batch(population)
                    scores = np.nan_to_num(scores, nan=1.0)

                    success_idx = np.where(scores < 0.5)[0]
                    if len(success_idx) > 0:
                        success_generation = generation + 1
                        break

                    inv_scores = 1.0 - scores

                    total = inv_scores.sum()
                    num_positive = np.count_nonzero(inv_scores)

                    if total <= 1e-12 or num_positive < 2:
                        selection_probs = np.ones_like(inv_scores) / len(inv_scores)
                    else:
                        selection_probs = inv_scores / total

                    indices = np.arange(len(population))
                    new_pop = []

                    new_pop.append(population[np.argmin(scores)])

                    while len(new_pop) < attack.population_size:
                        p1_idx, p2_idx = np.random.choice(indices, size=2, p=selection_probs, replace=False)
                        child = attack.crossover(population[p1_idx], population[p2_idx])
                        child = attack.mutate(child)
                        new_pop.append(child)

                    population = new_pop
                
                if success_generation == -1:
                    success_generation = attack.max_generations
                
                generations_list.append(success_generation)
            
            if generations_list:
                avg_generations = np.mean(generations_list)
                results[threshold] = avg_generations
                print(f"Dummy row threshold {threshold}: Average generations = {avg_generations:.2f}")
            else:
                results[threshold] = 0
                print(f"Dummy row threshold {threshold}: No successful attacks")
        
        return results

    def plot_dummy_row_results(self,results):
        """
        Create line plot showing the relationship between ECU-controlled dummy row threshold and attack performance.
        
        Shows how the amount of ECU-controlled modification space affects attack difficulty.
        
        Args:
            results: Dictionary mapping dummy row thresholds to average generations needed
        """
        thresholds = list(results.keys())
        avgs = [results[t] for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, avgs, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Dummy Row Threshold')
        plt.ylabel('Average Number of Generations')
        plt.title('Effect of Dummy Row Threshold on Spoofing Adversarial Attack Generations')
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.savefig('spoofing_dummy_rows_vs_generations.png')
        plt.close()
        
        print("Dummy row experiment plot saved as 'spoofing_dummy_rows_vs_generations.png'")

    def apply(self, cfg):
        """
        Main function that orchestrates the complete adversarial spoofing attack generation and evaluation process.

        Process:
        1. Load pre-trained IDS model for RPM/spoofing attacks
        2. Generate adversarial spoofing attack dataset (or load if exists)
        3. Evaluate model performance on adversarial examples
        4. Compute detailed performance metrics
        5. Create visualizations of misclassified examples
        6. Run parameter optimization experiments
        7. Generate performance analysis plots
        """
        dir_path     = cfg['dir_path']
        dataset_name = cfg['dataset_name']

        model_path  = "RPM_final_model.h5"
        attack_file = os.path.join(dir_path, "..", "datasets", dataset_name, "adversarial_spoofing_attack.npz")
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "Results", "attack_results", f"Spoof_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        attack = AdversarialSpoofAttack(
            model_path=model_path,
            population_size=100,    # Population size for genetic algorithm
            max_generations=75,     # Maximum evolution generations
            mutation_rate=0.2       # Mutation probability per individual
        )

        if os.path.exists(attack_file):
            print("Adversarial attack already exists. Loading from file...")
            try:
                data = np.load(attack_file)
                final_test = data['final_test']  # Adversarial/benign frames
                y_test = data['y_test']          # True labels
                x_test = data['x_test']          # Original frames
            except Exception as e:
                print(f"Error loading file: {e}")
                final_test, y_test, x_test, _ = attack.generate_adversarial_attack(dummy_row_threshold=1, max_frames=100)
                try:
                    np.savez(attack_file,
                            final_test=final_test,
                            y_test=y_test,
                            x_test=x_test)
                    print(f"Adversarial attack generated and saved to {attack_file}")
                except Exception as e:
                    print(f"Error saving file: {e}")
        else:
            try:
                print("Generating new adversarial spoofing attack dataset...")
                final_test, y_test, x_test, _ = attack.generate_adversarial_attack(dummy_row_threshold=1, max_frames=100)

                np.savez(attack_file,
                        final_test=final_test,
                        y_test=y_test,
                        x_test=x_test)
                print(f"Adversarial attack generated and saved to {attack_file}")
            except Exception as e:
                print(f"Error during attack generation or saving: {e}")

        print("Evaluating model performance on adversarial spoofing attack dataset...")
        test_loss, test_accuracy = attack.model.evaluate(final_test, y_test, verbose=1)

        with open(os.path.join(results_dir, "adversarial_spoof_test.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")

        print("Computing detailed performance metrics...")

        y_pred_prob = attack.model.predict(final_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

        FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # False Negative Rate
        ER = round((FP + FN) / (TN + FP + FN + TP), 4)                   # Error Rate
        precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0  # Precision
        recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0     # Recall (Sensitivity)
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0  # F1 Score

        GeneticAttack.plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_spoof_attack", normalize=True,
                            title='Normalized Confusion Matrix',
                            filename=os.path.join(results_dir, "confusion_matrix_adv_spoof_attack.png"))

        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

        with open(os.path.join(results_dir, "evaluation_metrics_spoof_adv.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            f.write(f"Error Rate (ER): {ER:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"Results for spoof adversarial attack (test set) saved")

        print("Creating visualizations of misclassified spoofing attack examples...")

        cnt = 0
        for i in range(len(y_test)):
            if cnt == 5:  # Limit to 5 examples to avoid clutter
                break

            if y_test[i] != y_pred[i] and y_test[i] == 1:
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Original Spoof Attack (True: {y_test[i]})")

                plt.subplot(1, 2, 2)
                plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Adversarial Spoof Attack (Predicted: {y_pred[i]})")

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"spoof_attack_comparison_{cnt}.png"))
                plt.close()
                print(f"Attack comparison saved as spoof_attack_comparison_{cnt}.png")
                cnt += 1
        
        print("\n===== RUNNING MUTATION RATE EXPERIMENT =====")
        mutation_results = GeneticAttack.run_mutation_rate_experiment(model_path, max_frames=20)
        GeneticAttack.plot_mutation_rate_results(mutation_results)
        
        print("\n===== RUNNING DUMMY ROW THRESHOLD EXPERIMENT =====")
        dummy_row_results = self.run_dummy_row_experiment(model_path, max_frames=20)
        plot_dummy_row_results(dummy_row_results)

