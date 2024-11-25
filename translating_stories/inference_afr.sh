#!/bin/bash

# Usage:
# sh inference_afr.sh [Path-To-Model] [src_lang] [tgt_lang] [batch_size] [input_file] [output_dir] [output_file]

MODEL="$1"
SRC="$2"
TGT="$3"
BATCH="$4"
SRCTEXT="$5"
OUTDIR="$6"
OUTPUT_FILE="$7"  # New parameter for the output file

# Paths for language pairs and dictionaries
LangPair="${MODEL}/lang_pairs_7g106.txt"
LangDict="${MODEL}/langs.txt"

# Encode the source text
python3 fairseq/scripts/spm_encode.py \
    --model "${MODEL}/sentencepiece.bpe.model" \
    --inputs "${SRCTEXT}" \
    --outputs "${OUTDIR}/spm.test.${SRC}" \
    --output_format piece

# Preprocess
python3 fairseq/fairseq_cli/preprocess.py \
    -s "${SRC}" --only-source \
    --testpref "${OUTDIR}/spm.test" \
    --destdir "${OUTDIR}" \
    --workers 16 \
    --srcdict "${MODEL}/dict.eng.txt"

# Create necessary symbolic links
ln -sf "${OUTDIR}/dict.${SRC}.txt" "${OUTDIR}/dict.${TGT}.txt"
ln -sf "${OUTDIR}/test.${SRC}-None.${SRC}.idx" "${OUTDIR}/test.${SRC}-${TGT}.${SRC}.idx"
ln -sf "${OUTDIR}/test.${SRC}-None.${SRC}.bin" "${OUTDIR}/test.${SRC}-${TGT}.${SRC}.bin"
ln -sf "${OUTDIR}/test.${SRC}-None.${SRC}.idx" "${OUTDIR}/test.${SRC}-${TGT}.${TGT}.idx"
ln -sf "${OUTDIR}/test.${SRC}-None.${SRC}.bin" "${OUTDIR}/test.${SRC}-${TGT}.${TGT}.bin"

# Run translation and save output to a unique file
CUDA_VISIBLE_DEVICES=0 python3 fairseq/fairseq_cli/generate.py "${OUTDIR}" \
  -s "${SRC}" -t "${TGT}" \
  --path "${MODEL}/checkpoint_best.pt" \
  --task 'translation_multi_simple_epoch' \
  --gen-subset 'test' \
  --batch-size "${BATCH}" \
  --beam '4' \
  --lenpen '1.0' \
  --remove-bpe 'sentencepiece' \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "${LangDict}" \
  --lang-pairs "${LangPair}" \
  > "${OUTDIR}/test.${SRC}-${TGT}.${OUTPUT_FILE%.ssw}.log"

# Extract the hypotheses to the specified output file
grep -P "^H" "${OUTDIR}/test.${SRC}-${TGT}.${OUTPUT_FILE%.ssw}.log" | sort -V | cut -f 3- > "${OUTDIR}/${OUTPUT_FILE}"

# Cleanup
rm "${OUTDIR}/dict."*.txt
rm "${OUTDIR}/spm.test.${SRC}"
rm "${OUTDIR}/test."*.idx
rm "${OUTDIR}/test."*.bin
rm "${OUTDIR}/test.${SRC}-${TGT}.${OUTPUT_FILE%.ssw}.log"
