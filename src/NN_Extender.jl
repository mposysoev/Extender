module NN_Extender

using TOML
using BSON
using BSON: @save, @load
using Flux
using Plots

struct NNparams
    structure::Vector{Int64}
    activations::Vector{String}
end

function readModel(filename)
    @load filename model
    return model
end

function readInputFile(filename="input.toml")
    settings = TOML.parsefile(filename)

    input_structure = settings["input"]["structure"]
    input_activations = settings["input"]["activations"]
    output_structure = settings["output"]["structure"]
    output_activations = settings["output"]["activations"]

    input_nn_params = NNparams(input_structure, input_activations)
    output_nn_params = NNparams(output_structure, output_activations)

    return (input_nn_params, output_nn_params)
end

function getNNStructureVec(model)
    structure = []
    for layer in model.layers
        push!(structure, size(layer.weight)[2])
    end
    push!(structure, size(model.layers[end].weight)[1])

    structure = convert(Vector{Int64}, structure)
    return structure
end

function setParamsToZero!(model)
    for p in Flux.params(model)
        p .= 0.0  # Broadcasting 0 to all elements of the parameter array
    end
end

function checkStructures(input::NNparams, output::NNparams)
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

function helloMessage(input::NNparams, output::NNparams)
    println("""
///////////////////////////////////////////////////////////////////////////////
                                Extender
///////////////////////////////////////////////////////////////////////////////
    """)
    println("Network would be changed from:")
    println("$(input.activations)")
    println("$(input.structure)")
    println("To:")
    println("$(output.activations)")
    println("$(output.structure)")
end

function checkMathModelAndInput(
    inputNN::NNparams,
    input_file_name,
    input_model_file,
    input_model,
)
    input_structure = getNNStructureVec(input_model)
    if inputNN.structure != input_structure
        println("WARNING:")
        println("Something wrong with structures.")
        println("Structure the actual model $(input_model_file) -> $(input_structure)")
        println("Doesn't match with structure $(inputNN.structure) from $(input_file_name)")
        error("Abort!")
    end
end

function initOutputModel(outputNN::NNparams)
    structure = outputNN.structure
    activations = outputNN.activations

    layers = []
    for i in 1:length(activations)
        push!(layers, Dense(structure[i], structure[i+1], getfield(Main, Symbol(activations[i]))))
    end

    model = Chain(layers...)
    return model
end

function copy_matrix_into!(dest, src, start_row::Int, start_col::Int)
    # Check if the source matrix can fit into the destination matrix at the specified position
    if start_row + size(src, 1) - 1 > size(dest, 1) || start_col + size(src, 2) - 1 > size(dest, 2)
        error("Source matrix does not fit into the destination matrix at the specified start position.")
    end

    # Copy values from source to destination
    for i in 1:size(src, 1)
        for j in 1:size(src, 2)
            dest[start_row+i-1, start_col+j-1] = src[i, j]
        end
    end
end

function copyNNWeigths(input_model, output_model)
    for (id, layer) in enumerate(input_model)
        copy_matrix_into!(output_model[id].weight, layer.weight, 1, 1)
        copy_matrix_into!(output_model[id].bias, layer.bias, 1, 1)
    end

    return output_model
end

function plot_model_parameters(model)
    for layer in model
        # Check if the layer has parameters (weights and biases)
        if hasmethod(Flux.params, Tuple{typeof(layer)})
            layer_params = Flux.params(layer)
            for p in layer_params
                # Assuming the parameter is a 2D array (for weights)
                if ndims(p) == 2
                    p_plot = heatmap(Array(p), title="Weights")
                    display(p_plot)
                # For biases or any 1D parameter, we convert them into a 2D array for the heatmap
                elseif ndims(p) == 1
                    p_plot = heatmap(reshape(Array(p), 1, length(p)), title="Biases")
                    display(p_plot)
                end
            end
        end
    end
end

function main()
    input_file_name = "input.toml"
    input_model_file = "model-pre-trained.bson"

    input_model = readModel(input_model_file)
    inputNN, outputNN = readInputFile(input_file_name)

    checkMathModelAndInput(inputNN, input_file_name, input_model_file, input_model)
    checkStructures(inputNN, outputNN)
    # TODO: check if output bigger
    # check len of neurons should be +1 of activations length

    helloMessage(inputNN, outputNN)

    output_model = initOutputModel(outputNN)
    setParamsToZero!(output_model)

    output_model = copyNNWeigths(input_model, output_model)

    @save "output_model.bson" output_model
    println(output_model)

end

end # module NN_Extender
