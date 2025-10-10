import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
import plotly.express as px


def evaluate(self, dataloader, texts, labels, batch_size=16):
        """
        Evaluate ensemble on a dataset

        Args:
            dataloader: Not used (we process texts directly)
            texts: List of text strings
            labels: Ground truth labels
            batch_size: Batch size for processing

        Returns:
            metrics: Dictionary with accuracy, F1, kappa
        """
        print("\n[ENSEMBLE] Evaluating ensemble model...")

        all_preds = []
        all_probs = []

        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]

            preds, probs = self.predict_batch(batch_texts)
            all_preds.extend(preds)
            all_probs.extend(probs)

            if (i + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(texts)} samples...")

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(labels, all_preds)
        macro_f1 = f1_score(labels, all_preds, average='macro')
        kappa = cohen_kappa_score(labels, all_preds)

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'cohen_kappa': kappa,
            'predictions': all_preds,
            'probabilities': all_probs
        }

        print(f"\n[ENSEMBLE] Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}")

        # Detailed classification report
        print(f"\n[ENSEMBLE] Classification Report:")
        print(classification_report(labels, all_preds,
                                   target_names=['Negative', 'Neutral', 'Positive'],
                                   digits=4))

        return metrics


def visualize_confusion_matrix(y_true, y_pred, output_path):
    """
    Visualize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=list(range(1, 11)),
                    y=list(range(1, 11)),
                    title="Confusion Matrix")

    fig.write_html(output_path)

    return cm
