from config import train, test, evaluate


def main():
    model = train()
    results = test(model)

    metrics = evaluate(
        results["label"],
        results["anomaly"]
    )

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
