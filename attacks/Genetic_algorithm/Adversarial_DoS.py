#!/usr/bin/env python3
"""
Adversarial DoS Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on DoS (Denial of Service) attack scenarios by modifying dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_dos_attack.py
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

class AdversarialDosAttack(GeneticAttack):
    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial DoS attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """
        
        super().__init__(model_path, file_path, population_size, max_generations, mutation_rate)
        self.original_dummy_rows = []
        
    def find_dummy_rows(self, frame):
        """
        Identify rows in a CAN frame that are completely zero (dummy rows).
        
        These dummy rows represent unused CAN message slots and are the only
        parts of the frame we can modify without destroying the attack payload.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            List of row indices that contain all zeros
        """
        return [i for i in range(29) if np.all(frame[i,:,0] == 0)]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original dummy rows.
        
        Mutation process:
        1. Select a random original dummy row with probability = mutation_rate
        2. Clear the entire row to ensure only one bit is set
        3. Set exactly one random bit (excluding priority bits 0-2) to 1
        
        This maintains the constraint that dummy rows can only contain
        sparse, single-bit modifications.
        
        Args:
            frame: 29x29x1 numpy array to mutate
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        if random.random() < self.mutation_rate and self.original_dummy_rows:
            row_idx = random.choice(self.original_dummy_rows)
            
            mutated[row_idx, :, 0] = 0
            
            bit_to_flip = random.randint(3, 28)
            mutated[row_idx, bit_to_flip, 0] = 1
            
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Crossover strategy:
        - Start with parent1 as the base (child inherits most characteristics)
        - For each original dummy row, randomly choose to inherit from parent2
        - Only swap rows that were originally dummy rows (preserves attack payload)
        
        This allows combining successful mutations from different parents
        while maintaining the integrity of the original attack structure.
        
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

    def generate_adversarial_attack(self, dummy_row_threshold=10, max_frames=7000):
        """
        Main genetic algorithm to generate adversarial DoS attacks.
        
        Process:
        1. Load and filter frames based on dummy row count
        2. Create balanced dataset (70% attack, 30% benign)
        3. For each attack frame with sufficient dummy rows:
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum dummy rows required to process a frame
            max_frames: Maximum total frames to include in final dataset
            
        Returns:
            tuple: (final_test, y_test, orig_frame, generations_needed)
                - final_test: Adversarial frames ready for evaluation
                - y_test: Corresponding labels for the frames
                - orig_frame: Original frames before adversarial modification
                - generations_needed: List of generations required per attack frame
        """
        final_test = []      # Adversarial/benign frames for evaluation
        orig_frame = []      # Original frames for comparison
        generations_needed = []  # Track genetic algorithm performance
        success_counter = 0
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count

        print(f"Using {attack_count} attack frames and {benign_count} benign frames")

        benign_indices = np.where(self.y_test == 0)[0][:benign_count]

        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")

        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(self.x_test[i])

        print("Starting genetic algorithm for adversarial attack generation...")

        attack_frames_processed = 0
        attack_indices = np.where(self.y_test == 1)[0]  # Find all attack frames

        for i in attack_indices:
            if attack_frames_processed >= attack_count:
                break

            dummy_rows = self.find_dummy_rows(self.x_test[i])

            if len(dummy_rows) <= dummy_row_threshold:
                continue  # Skip this frame
                final_test.append(self.x_test[i])
                orig_frame.append(self.x_test[i])
                attack_frames_processed += 1
            else:
                self.original_dummy_rows = dummy_rows.copy()
                frame_copy = self.x_test[i].copy()

                for row_idx in dummy_rows:
                    bit_to_flip = random.randint(3, 28)  # Avoid priority bits
                    frame_copy[row_idx, bit_to_flip, 0] = 1

                mut_rate = self.mutation_rate  # Save current mutation rate
                self.mutation_rate = 1.0       # Force mutation for population diversity
                population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
                self.mutation_rate = mut_rate  # Restore original mutation rate

                success_generation = -1  # Track when successful attack was found
                success_counter = 0
                for generation in range(self.max_generations):

                    # Batch predict all individuals at once
                    scores = self.calculate_confidence_batch(population)
                    scores = np.nan_to_num(scores, nan=1.0)

                    # Check for successful evasion
                    success_idx = np.where(scores < 0.5)[0]
                    if len(success_idx) > 0:
                        winner = success_idx[0]
                        print(f"Successful attack found in generation {generation+1}")
                        success_counter+=1
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
                    # print("Success counter : ", success_counter)

                if success_generation == -1:
                    best_idx = np.argmin(scores)
                    print(f"Best score achieved: {scores[best_idx]:.4f}")
                    final_test.append(population[best_idx])
                    orig_frame.append(self.x_test[i])
                    success_generation = self.max_generations

                generations_needed.append(success_generation)
                attack_frames_processed += 1
                print(f"  Frame {attack_frames_processed}/{attack_count} — Gen {success_generation}/{self.max_generations}")
        
        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def save_adversarial_attack(self, frame, output_file="adversarial_dos_attack.npy"):
        """
        Save a single adversarial attack frame to disk.
        
        Args:
            frame: 29x29x1 numpy array to save
            output_file: Output filename (NPY format)
        """
        np.save(output_file, frame)
        print(f"Adversarial attack saved to {output_file}")

    def apply(self, cfg):
        """
        Main function that orchestrates the complete adversarial DoS attack generation and evaluation process.

        Process:
        1. Load pre-trained IDS model for RPM/spoofing attacks
        2. Generate adversarial DoS attack dataset (or load if exists)
        3. Evaluate model performance on adversarial examples
        4. Compute detailed performance metrics
        5. Create visualizations of misclassified examples
        6. Run parameter optimization experiments
        7. Generate performance analysis plots
        """
        dir_path     = cfg['dir_path']
        dataset_name = cfg['dataset_name']

        attack_file = os.path.join(dir_path, "..", "datasets", dataset_name, "adversarial_DoS_attack.npz")

        print("Generating new adversarial DoS attack dataset...")
        final_test, y_test, x_test, _ = self.generate_adversarial_attack(dummy_row_threshold=10, max_frames=7000)
        
        np.savez(attack_file, 
                final_test=final_test, 
                y_test=y_test, 
                x_test=x_test)
        print(f"Adversarial attack generated and saved to {attack_file}.npz")

        
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "Results", "attack_results", f"DoS_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        print("  Evaluating model on adversarial dataset...")
        test_loss, test_accuracy = self.model.evaluate(final_test, y_test, verbose=0)

        with open(os.path.join(results_dir, "adversarial_DoS_test.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")

        y_pred_prob = self.model.predict(final_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

        FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # False Negative Rate
        ER = round((FP + FN) / (TN + FP + FN + TP), 4)                   # Error Rate
        precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0  # Precision
        recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0     # Recall (Sensitivity)
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0  # F1 Score
        ASR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # Attack Success Rate: adversarial attacks misclassified as Normal
        self.plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_DoS_attack", normalize=True,
                            title='Normalized Confusion Matrix',
                            filename=os.path.join(results_dir, "confusion_matrix_adv_DoS_attack.png"))

        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

        with open(os.path.join(results_dir, "evaluation_metrics_DoS_adv.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            f.write(f"Error Rate (ER): {ER:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"\n  Accuracy       : {test_accuracy:.4f}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
        print(f"  F1 Score       : {f1:.4f}")
        print(f"  FNR            : {FNR:.4f}")
        print(f"  Error Rate     : {ER:.4f}")
        print(f"  Attack Success rate : {ASR:.4f}")
        print(f"\n{report}")

        cnt = 0
        for i in range(len(y_test)):
            if cnt == 5:  # Limit to 5 examples to avoid clutter
                break

            if y_test[i] != y_pred[i] and y_test[i] == 1:
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Original DoS Attack (True: {y_test[i]})")

                plt.subplot(1, 2, 2)
                plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Adversarial DoS Attack (Predicted: {y_pred[i]})")

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"DoS_attack_comparison_{cnt}.png"))
                plt.close()
                cnt += 1

