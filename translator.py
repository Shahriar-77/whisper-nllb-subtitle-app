from transformers import NllbTokenizer, AutoModelForSeq2SeqLM


class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cuda"):
        # Had some issues with the fast_tokenizer
        self.tokenizer = NllbTokenizer.from_pretrained(model_name,use_Fast=False)
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def translate(self, text: str, src_lang="eng_Latn", tgt_lang="ben_Beng", max_length=512) -> str:
       
        # Set soruce language 
        self.tokenizer.src_lang = src_lang

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length).to(self.device)

        # Target BOS token
        bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        # Translate
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id
        )

        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

