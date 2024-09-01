using BSON: @save
using Flux

include("src/Extender.jl")
using .Extender

function main()
    if length(ARGS) == 0
        input_file_name = "input.toml"
    else
        input_file_name = ARGS[1]
    end

    input_nn_params, output_nn_params = read_input_file(input_file_name)
    general_params = read_general_params(input_file_name)

    input_model = load_model(input_nn_params.file_name)
    output_model = run_extending(input_model, input_nn_params, output_nn_params, general_params)

    model = nothing
    model = output_model
    @save output_nn_params.file_name model
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
