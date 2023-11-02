
from pathlib import Path
import sys

import numpy as np
sys.path.append(Path(__file__).parent.parent / "Real-Time-Voice-Cloning")
from encoder import audio
from encoder import inference as encoder

"""
    This file plots speaker embeddings from multiple speech files using the speaker encoder.

    See Real-Time-Voice-Cloning at https://github.com/CorentinJ/Real-Time-Voice-Cloning for details.
"""


if __name__ == "__main__":	
    encoder_model_fpath = Path("saved_models/default/encoder.pt")

    encoder.load_model(encoder_model_fpath)

    wav_fpath = "datasets/LibriSpeech/train-other-500/segment_1.wav"

    wav_files = Path("Real-Time-Voice-Cloning/datasets/LibriSpeech").glob('*.wav')
    wav_files = Path("").glob('*.wav')
    embed_fpath = "embeddings"

    embeds = {}
    for wav_fpath in wav_files:
        # wav = np.load(wav_fpath)
        wav = encoder.preprocess_wav(wav_fpath)

        # embed_utterance(wav)
        kwargs = dict()

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = encoder.compute_partial_slices(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        frames = audio.wav_to_mel_spectrogram(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        partial_embeds = encoder.embed_frames_batch(frames_batch)
        embeds[wav_fpath.name] = partial_embeds

    classes = list(embeds.keys())
    num_classes = len(classes)
    all_embeds = np.concatenate([em for em in embeds.values()])
    all_labels = np.concatenate([[classes.index(lab)]*emb.shape[0] for lab, emb in embeds.items()])

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300)  # Use 3 components for 3D
    embedding_3d = tsne.fit_transform(all_embeds)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print('about to plot')
    # Plot the t-SNE-transformed data with labels
    for class_idx in range(num_classes):
        class_indices = np.where(all_labels == class_idx)[0]
        color = 'red' if 'L' in classes[class_idx] else 'blue'
        ax.scatter(
            embedding_3d[class_indices, 0],
            embedding_3d[class_indices, 1],
            embedding_3d[class_indices, 2],
            label=f'Class {class_idx}',
            c=color
        )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.set_title('t-SNE of Embedding Vectors in 3D with Labels')
    ax.legend()
    plt.savefig('embeddinggraph.png')
    plt.show()

