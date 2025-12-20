from config import train, test, evaluate

model = train()
results = test(model)

metrics = evaluate(
    results["label"],
    results["anomaly"]
)

print(metrics
)