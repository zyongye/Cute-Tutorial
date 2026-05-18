import moe_gate_up_nvfp4_swiglu as _impl


for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

main = _impl.main


if __name__ == "__main__":
    main()
