"""Read-only capture of native evaluation tokens and environment results."""
import os
from pathlib import Path
from seen_common import protocol

class SeenEvaluationTrainer:
    def _capture_seen(self, batch, expected):
        nt = batch.non_tensor_batch
        response_ids = batch.batch['responses']
        response_mask = batch.batch['attention_mask'][:, -response_ids.shape[-1]:].bool()
        marker = self.tokenizer.encode('<RETHINK>', add_special_tokens=False)
        marker_id = marker[0] if len(marker) == 1 else None
        root = Path(os.environ['ALFWORLD_DATA']).resolve()
        for i in range(len(response_ids)):
            game = Path(str(nt['gamefile'][i])).resolve()
            task = game.relative_to(root).as_posix()
            assert task in expected
            ids = response_ids[i][response_mask[i]].cpu().tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            self._seen_rows.append({
                'task_id': task, 'traj_uid': str(nt['traj_uid'][i]),
                'turn_step': int(nt['turn_step'][i]), 'response_ids': ids,
                'response_tokens': len(ids), 'response_text': text,
                'protocol': protocol(text, ids, marker_id),
                'native_action_valid': bool(nt['is_action_valid'][i]),
                'episode_reward': float(nt['episode_rewards'][i]),
                'episode_length': int(nt['episode_lengths'][i])})
