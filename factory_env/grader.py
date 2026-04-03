def compute_score(total_reward, max_possible=20):
   score = total_reward / max_possible
   return max(0.0, min(1.0, score))