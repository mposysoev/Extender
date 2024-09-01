module Extender

using TOML
using BSON: @save, @load
using Flux
using Plots
using Statistics

export run_extending, read_input_file, read_general_params, NeuralNetParams, GeneralParams,
       load_model, save_model

struct NeuralNetParams
    file_name::String
    structure::Vector{Int64}
    activations::Vector{String}
    use_bias::Bool
    precision::String
end

struct GeneralParams
    set_weigths_to::Float64
    test_samples::Int64
end

function load_model(file_path::String)::Flux.Chain
    model = nothing
    @load file_path model
    return model
end

function save_model(file_path::String, model::Flux.Chain)
    @save file_path model
end

function read_input_file(filename::String = "input.toml")
    settings = TOML.parsefile(filename)
    input_settings = settings["input"]
    output_settings = settings["output"]

    input_nn_params = NeuralNetParams(
        input_settings["file_name"],
        input_settings["structure"],
        input_settings["activations"],
        input_settings["use_bias"],
        input_settings["precision"]
    )
    output_nn_params = NeuralNetParams(
        output_settings["file_name"],
        output_settings["structure"],
        output_settings["activations"],
        output_settings["use_bias"],
        output_settings["precision"]
    )

    return input_nn_params, output_nn_params
end

function get_nn_structure_vector(model::Flux.Chain)
    structure = []
    for layer in model.layers
        push!(structure, size(layer.weight)[2])
    end
    push!(structure, size(model.layers[end].weight)[1])

    structure = convert(Vector{Int64}, structure)
    return structure
end

function set_all_weigths_to_value(model::Flux.Chain, eps::Real)::Flux.Chain
    for p in Flux.params(model)
        p .= eps
    end

    return model
end

function check_structures(input::NeuralNetParams, output::NeuralNetParams)
    l1 = length(input.activations)
    l2 = length(output.activations)
    l3 = length(input.structure)
    l4 = length(output.structure)

    problem = false
    if (l1 != l2)
        println("ERROR:")
        println(
            "Length of input vector $(input.activations) doesn't match with $(output.activations)",
        )
        println("Amount of layers of neural network should be the same!")
        problem = true
    end
    if (l3 != l4)
        println("ERROR:")
        println(
            "Length of input vector $(input.structure) doesn't match with $(output.structure)",
        )
        println("Amount of layers of neural network should be the same!")
        problem = true
    end
    if (input.activations != output.activations)
        println("ERROR:")
        println(
            "Vectors of activation functions $(input.activations) doesn't match with $(output.activations)",
        )
        println("Amount of layers of neural network should be the same!")
        problem = true
    end

    if problem
        error("Abort!")
    end
end

function hello_message(input::NeuralNetParams, output::NeuralNetParams)
    println("""
///////////////////////////////////////////////////////////
                      Extender
///////////////////////////////////////////////////////////
    """)
    println("Neural Network would be changed from:")
    println("File: $(input.file_name)")
    println("Activations: $(input.activations)")
    println("Structure: $(input.structure)")
    println("Use bias: $(input.use_bias)")
    println("Precision: $(input.precision)")
    println("To:")
    println("Saved to file: $(output.file_name)")
    println("Activations: $(output.activations)")
    println("Structure: $(output.structure)")
    println("Use bias: $(output.use_bias)")
    println("Precision: $(output.precision)")
end

function validate_input_structure(input_nn_params::NeuralNetParams, input_model::Flux.Chain)
    input_structure = get_nn_structure_vector(input_model)
    if input_nn_params.structure != input_structure
        println("[ ERROR:")
        println("    Something wrong with structures.")
        println("    Structure of the model from file $(input_nn_params.file_name) -> $(input_structure)")
        println("    Doesn't match with the structure $(input_nn_params.structure) from $(input_nn_params.file_name)")
        error("Abort!")
    end
end

function init_output_model(output_nn_params::NeuralNetParams)
    structure = output_nn_params.structure
    activations = output_nn_params.activations

    layers = []
    for i in 1:length(activations)
        push!(layers,
            Dense(structure[i], structure[i + 1],
                getfield(Main, Symbol(activations[i])), bias = output_nn_params.use_bias))
    end

    model = Chain(layers...)

    if output_nn_params.precision == "f64"
        model = f64(model)
    elseif output_nn_params.precision == "f32"
        model = f32(model)
    elseif output_nn_params.precision == "f16"
        model = f16(model)
    end

    return model
end

function copy_matrix_into!(dest, src, start_row::Int, start_col::Int)
    # Check if the source matrix can fit into the destination matrix at the specified position
    if start_row + size(src, 1) - 1 > size(dest, 1) ||
       start_col + size(src, 2) - 1 > size(dest, 2)
        error(
            "Source matrix does not fit into the destination matrix at the specified start position.",
        )
    end

    # Copy values from source to destination
    for i in 1:size(src, 1)
        for j in 1:size(src, 2)
            dest[start_row + i - 1, start_col + j - 1] = src[i, j]
        end
    end
end

function copy_nn_weigths(
        input_model::Flux.Chain, output_model::Flux.Chain, input_nn_params::NeuralNetParams)
    for (id, layer) in enumerate(input_model)
        copy_matrix_into!(output_model[id].weight, layer.weight, 1, 1)
        if input_nn_params.use_bias
            copy_matrix_into!(output_model[id].bias, layer.bias, 1, 1)
        end
    end

    return output_model
end

function plot_model_parameters(model)
    for layer in model
        # Check if the layer has parameters (weights and biases)
        if hasmethod(Flux.params, Tuple{typeof(layer)})
            layer_params = Flux.params(layer)
            for p in layer_params
                # Define a custom color gradient with white at 0
                color_gradient = cgrad([:blue, :white, :red], [0.0, 0.5, 1.0], rev = false)
                # Set color limits to ensure 0 is always white
                color_limits = (-1.0, 1.0)
                # Assuming the parameter is a 2D array (for weights)
                if ndims(p) == 2
                    x_ticks = 1:size(p, 2)
                    y_ticks = 1:size(p, 1)
                    p_plot = heatmap(
                        Array(p),
                        title = "Weights",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (y_ticks, string.(y_ticks)),
                        c = color_gradient,
                        clims = color_limits,
                        aspect_ratio = 1,
                        size = (800, 800)
                    )
                    display(p_plot)
                    # For biases or any 1D parameter, we convert them into a 2D array for the heatmap
                elseif ndims(p) == 1
                    x_ticks = 1:length(p)
                    p_plot = heatmap(
                        reshape(Array(p), 1, length(p)),
                        title = "Biases",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (1, "1"),
                        c = color_gradient,
                        clims = color_limits,
                        aspect_ratio = 1,
                        size = (800, 800)
                    )
                    display(p_plot)
                end
            end
        end
    end
end

function plot_model_weight_differences(model1, model2)
    for (layer1, layer2) in zip(model1, model2)
        # Check if the layer has parameters (weights and biases)
        if hasmethod(Flux.params, Tuple{typeof(layer1)}) &&
           hasmethod(Flux.params, Tuple{typeof(layer2)})
            params1 = Flux.params(layer1)
            params2 = Flux.params(layer2)

            for (p1, p2) in zip(params1, params2)
                # Ensure both parameters are arrays
                p1 = Array(p1)
                p2 = Array(p2)

                # Pad the smaller parameter with zeros to match the size of the larger one
                if size(p1) != size(p2)
                    if size(p1, 1) > size(p2, 1) || size(p1, 2) > size(p2, 2)
                        p2 = padarray(p2, size(p1) .- size(p2), Val(:right))
                    elseif size(p2, 1) > size(p1, 1) || size(p2, 2) > size(p1, 2)
                        p1 = padarray(p1, size(p2) .- size(p1), Val(:right))
                    end
                end

                # Compute the difference between the parameters
                param_diff = p1 .- p2

                # Define a custom color gradient with white at 0
                color_gradient = cgrad([:blue, :white, :red], [0.0, 0.5, 1.0], rev = false)

                # Assuming the parameter is a 2D array (for weights)
                if ndims(param_diff) == 2
                    x_ticks = 1:size(param_diff, 2)
                    y_ticks = 1:size(param_diff, 1)
                    diff_plot = heatmap(
                        Array(param_diff),
                        title = "Weights Difference",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (y_ticks, string.(y_ticks)),
                        c = color_gradient
                    )
                    display(diff_plot)
                    # For biases or any 1D parameter, we convert them into a 2D array for the heatmap
                elseif ndims(param_diff) == 1
                    x_ticks = 1:length(param_diff)
                    diff_plot = heatmap(
                        reshape(Array(param_diff), 1, length(param_diff)),
                        title = "Biases Difference",
                        xticks = (x_ticks, string.(x_ticks)),
                        yticks = (1, "1"),
                        c = color_gradient
                    )
                    display(diff_plot)
                end
            end
        end
    end
end

# Helper function to pad arrays with zeros
function padarray(A, padsize, val)
    padded_array = zeros(eltype(A), size(A) .+ padsize)
    padded_array[1:size(A, 1), 1:size(A, 2)] .= A
    return padded_array
end

function set_last_layer_ones(output_model::Flux.Chain)
    copy_matrix_into!(output_model[end].weight, ones(size(output_model[end].weight)), 1, 1)
    return output_model
end

function run_test(
        input_model::Flux.Chain, output_model::Flux.Chain, input_nn_params::NeuralNetParams,
        output_nn_params::NeuralNetParams, general_params::GeneralParams)
    mul = 1
    n1 = input_nn_params.structure[1]
    n2 = output_nn_params.structure[1]

    differences_vector = zeros(general_params.test_samples)

    for i in 1:(general_params.test_samples)
        input_vector1 = rand(n1) .* mul
        v_2 = rand(n2 - n1) .* mul
        input_vector2 = vcat(input_vector1, v_2)
        result1 = input_model(input_vector1)
        result2 = output_model(input_vector2)

        difference = abs(result1[1] - result2[1]) / abs(result1[1])
        differences_vector[i] = difference
        mul *= -1
    end

    mean_diff = mean(differences_vector) * 100
    std_diff = std(differences_vector) * 100

    println()
    println("===========================================================")
    println("Results of comparison models")
    println("Mean relative difference: $(mean_diff)% ± $(std_diff)%")
    println("===========================================================")
    println()
end

function validate_structures_and_activations(
        input_nn_params::NeuralNetParams, output_nn_params::NeuralNetParams)
    if length(input_nn_params.structure) > length(output_nn_params.structure)
        println("Number of layers in input NN couldn't be bigger that in output")
        error("Check amount of layers")
    end

    if length(input_nn_params.activations) > length(output_nn_params.activations)
        println("Number of layers in input NN couldn't be bigger that in output")
        error("Check amount of layers")
    end

    if (length(input_nn_params.activations) + 1) != length(input_nn_params.structure)
        println("$(length(input_nn_params.activations)) + 1 != $(length(input_nn_params.structure))")
        error("The length of input arrays for structure and activations doesn't match")
    end

    if (length(output_nn_params.activations) + 1) != length(output_nn_params.structure)
        println("$(length(output_nn_params.activations)) + 1 != $(length(output_nn_params.structure))")
        error("The length of output arrays for structure and activations doesn't match")
    end

    for i in 1:length(input_nn_params.structure)
        if input_nn_params.structure[i] > output_nn_params.structure[i]
            println(
                "input: $(input_nn_params.structure[i]) > output: $(output_nn_params.structure[i]) in layer number $i.",
            )
            error("Layers in new neural net should be bigger or the same at least")
        end
    end

    for i in 1:(length(input_nn_params.structure) - 1)
        if output_nn_params.activations[i] != input_nn_params.activations[i]
            println(
                "$(input_nn_params.activations[i]) != $(output_nn_params.activations[i]) in layer number $i",
            )
            error(
                "Activation functions should be the same for corresponding layers for compatibility",
            )
        end
    end

    for i in (length(input_nn_params.structure) - 1):(length(output_nn_params.structure) - 1)
        if output_nn_params.activations[i] != "identity"
            println("$(output_nn_params.activations[i]) should be identity")
            error(
                "All additional layers should have IDENTITY activation function for compatibility",
            )
        end
    end

    if input_nn_params.structure[end] != output_nn_params.structure[end]
        println("WARNING: the sizes of the last layer should be the same.")
    end
end

function read_general_params(filename::String = "input.toml")::GeneralParams
    settings = TOML.parsefile(filename)
    input_settings = settings["general"]

    general_nn_params = GeneralParams(
        input_settings["set_weigths_to"], input_settings["test_samples"])

    return general_nn_params
end

function init_additional_layers(output_model::Flux.Chain, input_nn_params::NeuralNetParams,
        output_nn_params::NeuralNetParams, eps::Real)::Flux.Chain
    if length(input_nn_params.activations) < length(output_nn_params.activations)
        last_n = input_nn_params.structure[end]
        for i in (length(input_nn_params.activations) + 1):length(output_nn_params.activations)
            sizes = size(output_model[i].weight)
            ones_for_last_layer = ones(sizes)
            ones_for_last_layer[(last_n + 1):end] .= eps
            copy_matrix_into!(output_model[i].weight, ones_for_last_layer, 1, 1)
        end
    end

    return output_model
end

function validate(input_model::Flux.Chain, input_nn_params::NeuralNetParams,
        output_nn_params::NeuralNetParams)
    validate_input_structure(input_nn_params, input_model)
    validate_structures_and_activations(input_nn_params, output_nn_params)
end

function extend(input_model::Flux.Chain, input_nn_params::NeuralNetParams,
        output_nn_params::NeuralNetParams, general_params::GeneralParams)
    output_model = init_output_model(output_nn_params)
    output_model = set_all_weigths_to_value(output_model, general_params.set_weigths_to)
    output_model = copy_nn_weigths(input_model, output_model, input_nn_params)
    output_model = init_additional_layers(
        output_model, input_nn_params, output_nn_params, general_params.set_weigths_to)

    return output_model
end

function run_extending(input_model::Flux.Chain, input_nn_params::NeuralNetParams,
        output_nn_params::NeuralNetParams, general_params::GeneralParams)::Flux.Chain
    hello_message(input_nn_params, output_nn_params)

    validate(input_model, input_nn_params, output_nn_params)
    output_model = extend(input_model, input_nn_params, output_nn_params, general_params)

    run_test(input_model, output_model, input_nn_params, output_nn_params, general_params)

    return output_model
end
end # module Extender
