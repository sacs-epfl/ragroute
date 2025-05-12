import os

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

from ragroute.config import USR_DIR


class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        cache_path = os.path.join(USR_DIR, ".cache/torch/sentence_transformers", model_name_or_path)
        transformer_model = Transformer(model_name_or_path, cache_dir=cache_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]
