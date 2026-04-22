#!/usr/bin/env python3
"""
Adversarial Fuzzy Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on Fuzzy attack scenarios by modifying CAN frame bits to evade detection while
maintaining attack characteristics.

Key differences from DoS attacks:
- Fuzzy attacks can modify any bit in the frame (not limited to dummy rows)
- Uses bit-flipping mutations across the entire 29x29 frame
- Applies multiple mutations per frame for increased perturbation
- More aggressive crossover strategy (pixel-by-pixel inheritance)

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_fuzzy_attack.py
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

class AdversarialFuzzyAttack(GeneticAttack):

    def __init__(self, model_path, file_path ,population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial Fuzzy attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """
        
        super().__init__(model_path, file_path, population_size, max_generations, mutation_rate)
        
    def mutate(self, frame):
        """
        Apply mutation to a frame by randomly flipping bits across the entire frame.
        
        Fuzzy attack mutation strategy:
        - Unlike DoS attacks, fuzzy attacks can modify any part of the frame
        - Applies multiple bit flips per mutation operation for stronger perturbation
        - Uses bit-flipping (0->1, 1->0) rather than just setting bits to 1
        - More aggressive approach suitable for fuzzy attack characteristics
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        if random.random() < self.mutation_rate:
            for _ in range(3):  # Apply 3 random bit flips per mutation event
                row_idx = random.randint(0, 28)
                bit_to_flip = random.randint(0, 28)
                
                mutated[row_idx, bit_to_flip, 0] = 1 - mutated[row_idx, bit_to_flip, 0]
                
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform pixel-by-pixel crossover between two parent frames.
        
        Fuzzy attack crossover strategy:
        - More fine-grained than DoS attacks (pixel-level vs row-level)
        - Each bit position has 50% chance to inherit from either parent
        - Creates diverse offspring by mixing successful mutations at bit level
        - Suitable for fuzzy attacks where any bit can contribute to evasion
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining bit-level characteristics from both parents
        """
        mask = np.random.random((29, 29, 1)) < 0.5
        child = np.where(mask, parent2, parent1)
        return child

    def generate_adversarial_attack(self, max_frames=7000):
        """
        Main genetic algorithm to generate adversarial Fuzzy attacks.
        
        Process:
        1. Create balanced dataset (70% attack, 30% benign)
        2. Add benign frames directly (no modification needed)
        3. For each attack frame:
           - Apply initial random perturbations across the frame
           - Initialize genetic algorithm population
           - Evolve through generations using bit-level crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
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
        
        attack_indices = np.where(self.y_test == 1)[0][:attack_count]
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
        
        for idx, i in enumerate(attack_indices):
            
            frame_copy = self.x_test[i].copy()
            
            for row_idx in range(29):
                if random.random() < 0.5:
                    bit_to_flip = random.randint(3, 28)
                    frame_copy[row_idx, bit_to_flip, 0] = 1
            
            mut_rate = self.mutation_rate  # Save current mutation rate
            self.mutation_rate = 1         # Force mutation for all initial individuals
            
            population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
            
            self.mutation_rate = mut_rate
            
            success_generation = -1  # Track when successful attack was found
            
            for generation in range(self.max_generations):

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
            print(f"  Frame {idx+1}/{len(attack_indices)} — Gen {success_generation}/{self.max_generations}")

        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def apply(self, cfg):
        """
        Main function that orchestrates the complete adversarial fuzzy attack generation and evaluation process.

        Process:
        1. Load pre-trained IDS model for fuzzy attacks
        2. Generate adversarial fuzzy attack dataset (or load if exists)
        3. Evaluate model performance on adversarial examples
        4. Compute detailed performance metrics
        5. Create visualizations of misclassified examples
        6. Run mutation rate optimization experiments
        7. Generate performance analysis plots
        """
        dir_path     = cfg['dir_path']
        dataset_name = cfg['dataset_name']

        attack_file = os.path.join(dir_path, "..", "datasets", dataset_name, "adversarial_Fuzzy_attack.npz")
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "Results", "attack_results", f"Fuzzy_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        if os.path.exists(attack_file):
            print("Adversarial attack already exists. Loading from file...")
            try:
                data = np.load(attack_file)
                final_test = data['final_test']
                y_test = data['y_test']
                x_test = data['x_test']
            except Exception as e:
                print(f"Error loading file: {e}")
                final_test, y_test, x_test, _ = self.generate_adversarial_attack(max_frames=7000)
                np.savez(attack_file, final_test=final_test, y_test=y_test, x_test=x_test)
        else:
            print("Generating new adversarial Fuzzy attack dataset...")
            final_test, y_test, x_test, _ = self.generate_adversarial_attack(max_frames=50000)
            np.savez(attack_file, final_test=final_test, y_test=y_test, x_test=x_test)

        print("  Evaluating model on adversarial dataset...")
        test_loss, test_accuracy = self.model.evaluate(final_test, y_test, verbose=0)

        with open(os.path.join(results_dir, "adversarial_fuzzy_test.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")

        y_pred_prob = self.model.predict(final_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0
        ER = round((FP + FN) / (TN + FP + FN + TP), 4)
        precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
        recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        GeneticAttack.plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_fuzzy_attack", normalize=True,
                            title='Normalized Confusion Matrix',
                            filename=os.path.join(results_dir, "confusion_matrix_adv_fuzzy_attack.png"))

        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

        with open(os.path.join(results_dir, "evaluation_metrics_fuzzy_adv.txt"), "w") as f:
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
        print(f"\n{report}")

        cnt = 0
        for i in range(len(y_test)):
            if cnt == 5:
                break
            if y_test[i] != y_pred[i] and y_test[i] == 1:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Original Fuzzy Attack (True: {y_test[i]})")
                plt.subplot(1, 2, 2)
                plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Adversarial Fuzzy Attack (Predicted: {y_pred[i]})")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"fuzzy_attack_comparison_{cnt}.png"))
                plt.close()
                cnt += 1

