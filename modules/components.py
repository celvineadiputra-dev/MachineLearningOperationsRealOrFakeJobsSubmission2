def init_components():
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    # ExampleGen
    example_gen = CsvExampleGen(input_base=DATA_CLEAN, output_config=output)

    # StatisticsGen
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    # ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
    )

    # Trainer
    trainer = Trainer(
        module_file=os.path.abspath(TRAINER_MODULE_FILE),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'])
    )

    # ModelResolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')
    
    return (
        example_gen,
        statistics_gen,
        schema_gen
    )