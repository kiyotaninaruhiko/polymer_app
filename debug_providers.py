
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from providers.atompair import AtomPairFPProvider, TopologicalTorsionFPProvider
from providers.gnn_embed import GNNEmbedProvider
from providers.unimol import UniMolProvider

class MockRecord:
    def __init__(self, smiles):
        self.input_id = "test"
        self.input_smiles_raw = smiles
        self.smiles_normalized = smiles
        self.parse_status = "OK"

def debug_provider(provider_cls, name):
    print(f"\n--- Testing {name} ---")
    try:
        provider = provider_cls()
        print(f"Provider initialized. Version: {provider.version}")
        
        records = [MockRecord("c1ccccc1")] # Benzene
        params = {}
        for p in provider.params_schema():
            params[p.name] = p.default
            
        print(f"Running featurize with params: {params}")
        result = provider.featurize(records, params)
        
        print(f"Success count: {result.success_count}")
        print(f"Error count: {result.error_count}")
        
        if result.error_count > 0:
            print("Errors:")
            print(result.meta_df[['descriptor_error']])
        else:
            print("Success! Feature shape:", result.features_df.shape)
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


# Special debug for UniMol
def debug_unimol():
    print(f"\n--- Testing UniMol Deep Debug ---")
    try:
        from unimol_tools import UniMolRepr
        clf = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=False)
        smiles_list = ["CCO"]
        print("Running get_repr...")
        reprs = clf.get_repr(smiles_list)
        print(f"Type of reprs: {type(reprs)}")
        if isinstance(reprs, dict):
            print(f"Keys: {reprs.keys()}")
        elif isinstance(reprs, list):
            print(f"Length: {len(reprs)}")
            if len(reprs) > 0:
                print(f"First item type: {type(reprs[0])}")
        else:
            print(f"Content: {reprs}")
            
    except Exception as e:
        print(f"UniMol Debug Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # debug_provider(AtomPairFPProvider, "AtomPair")
    # debug_provider(TopologicalTorsionFPProvider, "TopologicalTorsion")
    # debug_provider(GNNEmbedProvider, "GNN")
    # debug_provider(UniMolProvider, "UniMol")
    debug_unimol()
