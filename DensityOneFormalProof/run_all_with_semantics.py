# run_all_with_semantics.py
import runpy, os, sys
import semantics_hooks

def run_file(path):
    print(f"\n$ python {os.path.basename(path)}")
    try:
        runpy.run_path(path, run_name="__main__")
        return True, ""
    except SystemExit as e:
        return (e.code == 0), f"SystemExit code={e.code}"
    except Exception as e:
        import traceback
        return False, traceback.format_exc()

def main():
    ok = semantics_hooks.install_hooks()
    paths = [
        "/mnt/data/kernel.py",
        "/mnt/data/number_theory.py",
        "/mnt/data/geom_series.py",
        "/mnt/data/lsi_base.py",
        "/mnt/data/parseval_derive.py",
        "/mnt/data/halasz_derive.py",
        "/mnt/data/mr_theorem.py",
        "/mnt/data/bilinear_sieve.py",
        "/mnt/data/heath_brown.py",
        "/mnt/data/type_I.py",
        "/mnt/data/type_II.py",
        "/mnt/data/final_combination.py",
    ]
    all_ok = ok
    for p in paths:
        ok2, msg = run_file(p)
        print("OK" if ok2 else "FAIL", p, msg)
        all_ok = all_ok and ok2
    print("\n=== SUMMARY ===")
    print("All semantic hooks installed OK:", ok)
    print("Full pipeline OK:", all_ok)
    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
