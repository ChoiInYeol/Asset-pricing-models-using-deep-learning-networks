# 대충 좋은 에포크 입력하는 코드
n_factors = 6
units = 8
batch_size = 64 # 256
first_epoch = 180
last_epoch = 210


predictions = []
for epoch in tqdm(list(range(first_epoch, last_epoch))):
    epoch_preds = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(data)):
        X1_train, X2_train, y_train, X1_val, X2_val, y_val = get_train_valid_data(
            data, train_idx, val_idx
        )

        train_gen = DataGenerator(X1_train, X2_train, y_train, batch_size=batch_size)
        val_gen = DataGenerator(X1_val, X2_val, y_val, batch_size=batch_size)
        model = make_model(hidden_units=units, n_factors=n_factors)
        model.fit_generator(
            train_gen,
            validation_data=val_gen,
            epochs=epoch,
            verbose=0,
            shuffle=True,
            callbacks=[ClearMemory(), early_stop],
        )
        epoch_preds.append(
            pd.Series(
                model.predict_generator(
                    val_gen, callbacks=[ClearMemory(), early_stop]
                ).reshape(-1),
                index=y_val.stack().index,
            ).to_frame(epoch)
        )

    predictions.append(pd.concat(epoch_preds))

predictions_combined = pd.concat(predictions, axis=1).sort_index()
predictions_combined.to_hdf(results_path / "predictions.h5", "predictions")