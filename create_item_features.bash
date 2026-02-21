image_model=google/siglip-base-patch16-224
text_model=BAAI/bge-base-en-v1.5

#category=toys
#uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
#uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model

category=sports
uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
#uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model

category=beauty
uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode image --model-name $image_model
#uv run python -m seqrec.main_extract_features --dataset-path dataset/$category --mode text --model-name $text_model