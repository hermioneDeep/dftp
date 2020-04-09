import sys
training_distributed=sys.modules[sys.modules[tf.keras.models.Model.__module__].training_distributed.fit_distributed.__module__]
  
def tpu_init_train_model(model,x,y):
  
  #placeholders used to feed the data
  x_ph, y_ph = tf.placeholder(tf.float32, shape=x.shape), tf.placeholder(tf.float32, shape=y.shape)
  dataset = tf.data.Dataset.from_tensor_slices((x_ph, y_ph)).batch(x.shape[0]).repeat()

  current_strategy = model._distribution_strategy
  
  #iterator creation
  with current_strategy.scope():
    iterator = current_strategy.make_dataset_iterator(dataset)
    init_op = training_distributed.control_flow_ops.group(iterator.initialize())
    if not tf.executing_eagerly():
        training_distributed.K.get_session((init_op,)).run(init_op, feed_dict={x_ph: x, y_ph: y})

  scope = training_distributed.distributed_training_utils.distributed_scope(
      strategy=current_strategy, learning_phase=1)
  scope.__enter__()

  out_labels = model.metrics_names or []

  step_fn = training_distributed._make_train_step_fn(model, training_distributed.ModeKeys.TRAIN, current_strategy,
                                out_labels)

  # Add initial dummy values for loss and other metric tensors.
  initial_loop_values = {}
  initial_loop_values['loss'] = training_distributed.constant_op.constant(1e7)
  for name in model.metrics_names[1:]:
    tensor = model._all_metrics_tensors[name]
    initial_loop_values[name] = training_distributed.array_ops.zeros(tensor.shape, tensor.dtype)
    
  ctx = current_strategy.extended.experimental_run_steps_on_iterator(
      step_fn, iterator, iterations=1,
      initial_loop_values=initial_loop_values)
  train_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  if model._compile_distribution:
    training_distributed.distributed_training_utils._copy_weights_to_distributed_model(model, training_distributed.ModeKeys.TRAIN)
  
  training_distributed.distributed_training_utils._reset_metrics(model)
  
  model._tpu_train_op=train_op
  model._tpu_train_output_tensors=output_tensors
  model._tpu_train_iterator=iterator
  model._tpu_train_iterator_init_op = init_op
  model._tpu_train_x_ph=x_ph
  model._tpu_train_y_ph=y_ph
  
  scope.__exit__(None, None, None)
  
def tpu_train_on_batch(model,x,y):
  
  if not (hasattr(model, '_tpu_train_op')):
    tpu_init_train_model(model,x,y)

  #TRAINING
  scope = training_distributed.distributed_training_utils.distributed_scope(
      strategy=model._distribution_strategy, learning_phase=1)
  scope.__enter__()
  
  if not tf.executing_eagerly():
    training_distributed.K.get_session((model._tpu_train_iterator_init_op,)).run(model._tpu_train_iterator_init_op, feed_dict={model._tpu_train_x_ph: x, model._tpu_train_y_ph: y})

  train_tensors=[model._tpu_train_op, model._tpu_train_output_tensors]
  _, outputs = training_distributed.K.get_session(train_tensors).run(train_tensors)
    
  scope.__exit__(None, None, None)
  
  outputs=list(outputs.values())
  if len(outputs)==1:
    outputs=outputs[0]
  
  return outputs