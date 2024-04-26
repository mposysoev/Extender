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
        push!(layers, Dense(structure[i], structure[i+1], getfield(Main, Symbol(activations[i])), bias=false))
    end

    model = Chain(layers...)
    model = fmap(f64, model)
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
        # copy_matrix_into!(output_model[id].bias, layer.bias, 1, 1)
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

function setLastLayerOnes(output_model)
    copy_matrix_into!(output_model[end].weight, ones(size(output_model[end].weight)), 1, 1)
    return output_model
end

function test(input_model, output_model, inputParams::NNparams, outputParams::NNparams)
    mul = 1
    n1 = inputParams.structure[1]
    n2 = outputParams.structure[1]

    for i in 1:1000
        input_vector1 = rand(n1) .* mul
        v_2 = rand(n2 - n1) .* mul
        input_vector2 = vcat(input_vector1, v_2)
        result1 = input_model(input_vector1)
        result2 = output_model(input_vector2)
        if abs(result1[1] - result2[1]) > 0.00001
            println("❌ Failed test!")
            println("Input model result: $(result1[1])")
            println("Output model result: $(result2[1])")
            println("Difference: $(abs(result1[1] - result2[1]))")
        end
        mul *= -1
    end
end

function checkLayersSizes(inputNN::NNparams, outputNN::NNparams)
    if length(inputNN.structure) > length(outputNN.structure)
        println("Number of layers in input NN couldn't be bigger that in output")
        error("Check amount of layers")
    end

    if length(inputNN.activations) > length(outputNN.activations)
        println("Number of layers in input NN couldn't be bigger that in output")
        error("Check amount of layers")
    end

    if (length(inputNN.activations) + 1) != length(inputNN.structure)
        println("$(length(inputNN.activations)) + 1 != $(length(inputNN.structure))")
        error("The length of input arrays for structure and activations doesn't match")
    end

    if (length(outputNN.activations) + 1) != length(outputNN.structure)
        println("$(length(outputNN.activations)) + 1 != $(length(outputNN.structure))")
        error("The length of output arrays for structure and activations doesn't match")
    end


    for i in 1:length(inputNN.structure)
        if inputNN.structure[i] > outputNN.structure[i]
            println("input: $(inputNN.structure[i]) > output: $(outputNN.structure[i]) in layer number $i.")
            error("Layers in new neural net should be bigger or the same at least")
        end
    end

    for i in 1:(length(inputNN.structure)-1)
        if outputNN.activations[i] != inputNN.activations[i]
            println("$(inputNN.activations[i]) != $(outputNN.activations[i]) in layer number $i")
            error("Activation functions should be the same for corresponding layers for compatibility")
        end
    end

    for i in (length(inputNN.structure)-1):(length(outputNN.structure)-1)
        if outputNN.activations[i] != "identity"
            println("$(outputNN.activations[i]) should be identity")
            error("All additional layers should have IDENTITY activation function for compatibility")
        end
    end

    if inputNN.structure[end] != outputNN.structure[end]
        println("WARNING: the sizes of the last layer should be the same. (or not)")
    end
end

function main(input_model_file)
    input_file_name = "input.toml"

    input_model = readModel(input_model_file)
    inputNN, outputNN = readInputFile(input_file_name)

    checkMathModelAndInput(inputNN, input_file_name, input_model_file, input_model)
    helloMessage(inputNN, outputNN)

    checkLayersSizes(inputNN, outputNN)

    output_model = initOutputModel(outputNN)
    setParamsToZero!(output_model)
    output_model = copyNNWeigths(input_model, output_model)

    if length(inputNN.activations) < length(outputNN.activations)
        last_n = inputNN.structure[end]
        println("Adding additional layers")
        for i in (length(inputNN.activations)+1):length(outputNN.activations)
            sizes = size(output_model[i].weight)
            ones_for_last_layer = ones(sizes)
            ones_for_last_layer[(last_n+1):end] .= 0
            copy_matrix_into!(output_model[i].weight, ones_for_last_layer, 1, 1)
        end
    end

    test(input_model, output_model, inputNN, outputNN)

    @save "output_model.bson" output_model
end
end # module NN_Extender
