from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

import numpy as np
import transformers

logger = logging.getLogger(__name__)


@DatasetReader.register("wordpiece_ud")
class UDWordpieceReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]

                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]

                instance = self.text_to_instance(words, pos_tags, list(zip(tags, heads)))
                if instance:
                    yield instance
                else:
                    continue

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
    ) -> Instance:

        """
        Parameters
        ----------
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        # model_type = "bert" if isinstance(self.tokenizer.tokenizer, transformers.BertTokenizer) else "xlmr"
        model_type = "bert"

        words = [i.replace(' ', '') for i in words]

        self.tokenizer._add_special_tokens = True
        bos, eos = self.tokenizer.tokenize("")
        self.tokenizer._add_special_tokens = False

        wordpieces = [bos]
        for i in words:
            current = self.tokenizer.tokenize(i)
            for j in current[1:]:
                if model_type == 'bert':
                    j.text = '##' + j.text if not j.text.startswith('##') else j.text
            wordpieces.extend(current)
        wordpieces.append(eos)

        if len(wordpieces) >= 256:
            logger.warning(f"Too large: dropping {' '.join(words)}")
            return None

        # build map
        offsets = []
        for n, piece in enumerate(wordpieces):
            if model_type == "bert":
                if not piece.text.startswith('##'):
                    offsets.append(n)
            elif model_type == "xlmr":
                if piece.text.startswith('▁'): # NOT an underscore!
                    offsets.append(n)

        offsets = offsets[1:-1] if model_type == 'bert' else offsets

        if len(offsets) != len(words):
            logger.warning(f'offset fail: dropping {" ".join(words)}')
            return None

        tokens = [Token(t) for t in words]

        wordpiece_field = TextField(wordpieces, self._token_indexers)
        offset_field = ArrayField(np.array(offsets), dtype=np.long)

        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = wordpiece_field
        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")
        fields["offsets"] = offset_field
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies], text_field, label_namespace="head_index_tags"
            )

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags, "wordpieces": wordpieces})
        return Instance(fields)
