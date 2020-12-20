#
# metrics = {'deepsets': {'acc' :[], 'mae' :[], 'mse' :[]}, 'lstm': {'acc' :[], 'mae' :[], 'mse' :[]}, 'gru': {'acc' :[], 'mae' :[], 'mse' :[]}}
#
# lengths = range(min_test_length, max_test_length, step_test_length)
# for l in lengths:
#     print('Evaluating at length: ', l)
#     K.clear_session()
#
#     # generate test data
#     Y, sum_Y = gen_test_data(num_test_examples, l)
#
#     # model
#     model = get_deepset_model(l)
#
#     # load weights
#     for i, idx in enumerate([1 ,2 ,4]):
#         model.get_layer(index=idx).set_weights(deep_we[i])
#
#     # prediction
#     preds = model.predict(Y, batch_size=128, verbose=1)
#     metrics['deepsets']['acc'].append(1. 0 *np.sum(np.squeeze(np.round(preds) )= =sum_Y ) /len(sum_Y))
#     metrics['deepsets']['mae'].append(np.sum(np.abs(np.squeeze(preds ) -sum_Y) ) /len(sum_Y))
#     metrics['deepsets']['mse'].append(np.dot(np.squeeze(preds ) -sum_Y, np.squeeze(preds ) -sum_Y ) /len(sum_Y))
#
#     # model
#     model = get_lstm_model(l)
#
#     # load weights
#     for i, idx in enumerate([1 ,2 ,3]):
#         model.get_layer(index=idx).set_weights(lstm_we[i])
#
#     # prediction
#     preds = model.predict(Y, batch_size=128, verbose=1)
#     metrics['lstm']['acc'].append(1. 0 *np.sum(np.squeeze(np.round(preds) )= =sum_Y ) /len(sum_Y))
#     metrics['lstm']['mae'].append(np.sum(np.abs(np.squeeze(preds ) -sum_Y) ) /len(sum_Y))
#     metrics['lstm']['mse'].append(np.dot(np.squeeze(preds ) -sum_Y, np.squeeze(preds ) -sum_Y ) /len(sum_Y))
#
#     # model
#     model = get_gru_model(l)
#
#     # load weights
#     for i, idx in enumerate([1 ,2 ,3]):
#         model.get_layer(index=idx).set_weights(gru_we[i])
#
#     # prediction
#     preds = model.predict(Y, batch_size=128, verbose=1)
#     metrics['gru']['acc'].append(1. 0 *np.sum(np.squeeze(np.round(preds) )= =sum_Y ) /len(sum_Y))
#     metrics['gru']['mae'].append(np.sum(np.abs(np.squeeze(preds ) -sum_Y) ) /len(sum_Y))
#     metrics['gru']['mse'].append(np.dot(np.squeeze(preds ) -sum_Y, np.squeeze(preds ) -sum_Y ) /len(sum_Y))