# rs_model/__main__.py
# Позволяет запускать `python -m rs_model`, но основная точка — backtests/wf_is_rs.py
if __name__ == "__main__":
    from backtests.wf_is_rs import main
    main()