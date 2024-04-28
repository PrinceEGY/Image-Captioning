from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, ds, feature_extractor):
        self.model = model
        self.ds = ds
        self.feature_extractor = feature_extractor

    def evaluate_bleu_greedy(
        self,
        temperatures=(0, 0.2, 0.5, 1),
        output_file="results/bleu_results_greedy.txt",
    ):
        hypotheses = [[] for _ in temperatures]
        references = []
        ds = self.ds.map(
            lambda x, y: (self.feature_extractor(x), y), tf.data.AUTOTUNE
        ).batch(64)
        with open(output_file, "w+") as f:  # Open file for writing
            for images, caps in tqdm(ds, desc="Evaluating..."):
                for idx, temp in enumerate(temperatures):
                    pred_caps = self.model.greedy_gen(
                        images, temperature=temp, max_len=30
                    )
                    hypotheses[idx] += [cap.split() for cap in pred_caps]
                references += [
                    [sen.decode("utf-8").split() for sen in cap.numpy().tolist()]
                    for cap in caps
                ]

            for idx, temp in enumerate(temperatures):
                f.write("===== Evaluation with temperature " + str(temp) + " =====\n")
                f.write(
                    "Cumulative 1-gram: %f\n"
                    % corpus_bleu(references, hypotheses[idx], weights=(1, 0, 0, 0))
                )
                f.write(
                    "Cumulative 2-gram: %f\n"
                    % corpus_bleu(references, hypotheses[idx], weights=(0.5, 0.5, 0, 0))
                )
                f.write(
                    "Cumulative 3-gram: %f\n"
                    % corpus_bleu(
                        references, hypotheses[idx], weights=(0.33, 0.33, 0.33, 0)
                    )
                )
                f.write(
                    "Cumulative 4-gram: %f\n"
                    % corpus_bleu(
                        references, hypotheses[idx], weights=(0.25, 0.25, 0.25, 0.25)
                    )
                )
            f.write("\n")
            f.seek(0)
            print(f.read())

        print("Results saved to", output_file)

    def evaluate_bleu_beam(
        self,
        kbeams=1,
        output_file="results/bleu_results_beams.txt",
    ):
        hypotheses = []
        references = []
        ds = self.ds.map(
            lambda x, y: (self.feature_extractor(x), y), tf.data.AUTOTUNE
        ).batch(1)

        with open(output_file, "w+") as f:
            for images, caps in tqdm(ds, desc="Evaluating..."):
                pred_caps = self.model.beam_search_gen(
                    images, Kbeams=kbeams, max_len=30
                )[0]
                hypotheses += [pred_caps.split()]
                references += [
                    [sen.decode("utf-8").split() for sen in cap.numpy().tolist()]
                    for cap in caps
                ]

            f.write("===== Evaluation using " + str(kbeams) + " beams =====\n")
            f.write(
                "Cumulative 1-gram: %f\n"
                % corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
            )
            f.write(
                "Cumulative 2-gram: %f\n"
                % corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
            )
            f.write(
                "Cumulative 3-gram: %f\n"
                % corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
            )
            f.write(
                "Cumulative 4-gram: %f\n"
                % corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
            )
            f.write("\n")
            f.seek(0)
            print(f.read())

        print("Results saved to", output_file)
