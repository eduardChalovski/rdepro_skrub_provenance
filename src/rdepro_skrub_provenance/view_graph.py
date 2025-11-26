from __future__ import annotations

import skrub
from rdepro_skrub_provenance.data.dataset import load_simple_dataset_pandas


def main() -> None:
    df = load_simple_dataset_pandas()

    t = skrub.X(df)  

    filtered = t[t["age"] > 30]
    projected = filtered[["name", "city"]]

    print("Résultat du pipeline :")
    print(projected.skb.eval()) 

    drawing = projected.skb.draw_graph()

    drawing.open()

if __name__ == "__main__":
    main()
