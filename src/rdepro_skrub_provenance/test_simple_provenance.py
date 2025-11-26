from rdepro_skrub_provenance.data.dataset import load_simple_dataset_pandas
from rdepro_skrub_provenance.simple_ops import (
    rename_column,
    rename_column_with_prov,
)
from rdepro_skrub_provenance.provenance_utils import with_provenance


def main():
    df = load_simple_dataset_pandas()
    print("=== DataFrame original ===")
    print(df)

    df_prov = with_provenance(df, source_name="simple")
    print("\n=== DataFrame + provenance ===")
    print(df_prov)

    df_renamed = rename_column_with_prov(df_prov, "age", "Age")
    print("\n=== Après renommage (avec provenance) ===")
    print(df_renamed)


if __name__ == "__main__":
    main()
