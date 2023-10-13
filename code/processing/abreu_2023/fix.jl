# List files
files = Glob.glob("./output/*/*.csv")

# Select columns that indicate the dataset
df_idx = df_counts[:, 13:end]

# Enumerate the columns (needed to run things in parallel smoothly)
n_col = size(df_idx, 2)

##

counter = 0
# Threads.@threads 
for col = 1:n_col

    # Extract data
    global data = df_counts[df_idx[:, col], 1:12]

    # Compile directory name
    out_dir = "./output/$(first(data.condition))_$(names(df_idx)[col])"

    file = Glob.glob("$(out_dir)/*3000*csv")

    global df_advi = CSV.read(file, DF.DataFrame)

    # Extract unique replicates WITHOUT SORTING
    global rep_unique = ["$x" for x in unique(data.rep)]

    # Dictionary from rep_unique to sort(rep_unique)
    global rep_dict = Dict(zip(rep_unique, sort(rep_unique)))

    # Define replicates
    global reps = deepcopy(df_advi.rep)

    for (i, r) in enumerate(df_advi.rep)
        if r ≠ "N/A"
            reps[i] = rep_dict[r]
        end #if
    end # for

    if any(reps .≠ df_advi.rep)
        println(file)
        counter += 1
        println(counter)
        # df_advi = df_advi[:, DF.Not(:rep)]
        # df_advi[!, :rep] = reps
        # CSV.write(first(file), df_advi)
    end

end # for

##