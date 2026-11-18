import json
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ClaimLevelVerifier:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
            
        self.retriever = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        nli_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

    def _process_data2txt(self, context: str) -> str:
        try:
            data = json.loads(context)
            if isinstance(data, dict):
                return ". ".join([f"{k}: {v}" for k, v in data.items()]) + "."
        except json.JSONDecodeError:
            pass
        return context

    def _split_claims(self, answer: str) -> list:
        doc = self.nlp(answer)
        claims = []
        for sent in doc.sents:
            text = sent.text.strip()
            if text.lower().startswith("here is") or text.endswith(":"): continue
            if len(text) > 5: claims.append(text)
        return claims

    def _chunk_context(self, context: str, chunk_size=256, stride=128) -> list:
        tokens = self.tokenizer.tokenize(context)
        chunks = []
        for i in range(0, max(1, len(tokens)), chunk_size - stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk_tokens))
            if i + chunk_size >= len(tokens): break
        return chunks

    def verify(self, context: str, answer: str, task: str = "qa") -> dict:
        if task.lower() == "data2txt":
            context = self._process_data2txt(context)
            
        claims = self._split_claims(answer)
        if not claims: return {"claims": []}
            
        chunks = self._chunk_context(context)
        if not chunks:
            return {"claims": [{"text": c, "p_entail": 0.0, "p_contra": 1.0} for c in claims]}

        chunk_embeddings = self.retriever.encode(chunks, convert_to_tensor=True)
        claim_embeddings = self.retriever.encode(claims, convert_to_tensor=True)
        cos_scores = util.cos_sim(claim_embeddings, chunk_embeddings)
        
        pairs, pair_map = [], []
        for claim_idx, claim in enumerate(claims):
            top_results = torch.topk(cos_scores[claim_idx], k=min(3, len(chunks)))
            for chunk_idx in top_results.indices:
                pairs.append((chunks[chunk_idx], claim))
                pair_map.append(claim_idx)

        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            
        ent_idx = self.nli_model.config.label2id.get("entailment", 0)
        con_idx = self.nli_model.config.label2id.get("contradiction", 2)
        ent_probs, con_probs = probs[:, ent_idx].cpu().tolist(), probs[:, con_idx].cpu().tolist()

        claim_scores = {}
        for c_idx, e_score, c_score in zip(pair_map, ent_probs, con_probs):
            if c_idx not in claim_scores:
                claim_scores[c_idx] = {"p_entail": e_score, "p_contra": c_score}
            else:
                claim_scores[c_idx]["p_entail"] = max(claim_scores[c_idx]["p_entail"], e_score)
                claim_scores[c_idx]["p_contra"] = max(claim_scores[c_idx]["p_contra"], c_score)

        final_claims = []
        for i in range(len(claims)):
            final_claims.append({
                "text": claims[i], 
                "score": claim_scores.get(i, {"p_entail": 0.0})["p_entail"],
                "p_entail": claim_scores.get(i, {"p_entail": 0.0})["p_entail"], 
                "p_contra": claim_scores.get(i, {"p_contra": 0.0})["p_contra"]
            })
        
        # Calculate overall document score and test schema compliance
        p_contras = [c.get("p_contra", 0.0) for c in final_claims]
        overall_score = max(p_contras) if p_contras else 0.0

        return {
            "score": overall_score,
            "hallucinated": overall_score >= 0.5,
            "claims": final_claims
        }