import json
import os
from typing import Dict


def dump_results_to_json(model_name: str, beam_size: int, lang_stats: Dict[str, float], out_path: str) -> None:
    file_name = os.path.join(out_path, model_name+str(beam_size)+".json")
    out_json = {
        "model_name": model_name,
        "beam_size": beam_size,
        "scores": dict(),
    }
    out_json['scores'].update(lang_stats)
    print(out_json)
    print("saving f{file_name}")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(out_json, f)
    print("saving done")


if __name__ == '__main__':
    dump_results_to_json(
        "ss",
        1,
        {"s":1},
        ""
    )
