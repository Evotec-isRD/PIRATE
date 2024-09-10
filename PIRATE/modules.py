# Import libraries
import pirate
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def main_page() -> str:
    """
    main_page runs the code/styling for the e3p main page in the streamlit app. The code
    will prompt the user to select the operation mode. This choice is returned as a string
    stored in the variable choice.

    :return: A string indicating use mode
    """

    # Title
    st.title("PIRATE: Yo Ho!")

    choice = st.radio(label='Mode Selection',
                      options=["Basic Sequence Analysis", "PIRATE Directed Evolution"], horizontal=True)

    return choice


def make_plot(predictions):
    """
    Function that plots the predictions for each residue as a time-series
    """
    arr = np.array(predictions)
    fig, ax = plt.subplots()

    plt.title("Probability of Disorder per Residue", loc="center")
    plt.xlabel("Residue")
    plt.ylabel("Probability of Disorder")
    ax.plot(arr)

    return fig


def basic_sequence_analysis() -> None:
    """
    basic_sequence_analysis contains the streamlit code for the basic sequence analysis
    """

    st.header("Basic Sequence Analysis")

    sequence = st.text_input(label="Please input sequence to analyze for disordered regions")
    sequence = sequence.replace(" ", "")

    with st.form(key="my_form_to_submit"):
        submit_button = st.form_submit_button(label="submit")

    if submit_button:
        predictions = pirate.PirateEvolutionTool(sequence=sequence).basic_disorder_prediction()
        # Plot disorder
        fig = make_plot(predictions)
        st.pyplot(fig)


def directed_evolution() -> None:
    """
    directed_evolution contains the streamlit code for the PIRATE directed evolution functionality
    """
    st.header("PIRATE Directed Evolution")

    out_file_name = st.text_input(label="Please input a name for the output CSV file. If none is provided, "
                                        "we'll just name the file 'pirate_sequences.csv'.")

    sequence = st.text_input(label="Please input sequence to analyze for disordered regions. The sequence "
                                   "length may not exceed 2048 residues. Please note that pLDDT scoring "
                                   "may be slightly degraded with sequences longer than 400 residues since "
                                   "we won't be able to submit the full sequence to the ESMFold API.")
    sequence = sequence.replace(" ", "")
    max_residues_to_alter = int(st.number_input(label="Max number of residues to alter", min_value=1))
    max_iterations = int(st.number_input(label="Max number of iterations. If left at 0, the AI will keep going "
                                               "until the desired number of mutations is reached", min_value=0))
    protected_residues = st.text_input(label="If there are residues you do not want altered, enter them "
                                             "here as comma-separated integers.", value=None)
    st.markdown("<span style='color:red'> Please note that, for technical reasons, PIRATE cannot be be used to "
                "optimize the first eleven or last eleven residues in the sequence. If you believe that these are "
                "disordered, you can leave them in the sequence during the evolution process and then manually truncate"
                " them later. The AI is programmed to ignore these residues. Because terminal residues are not "
                "optimized, PIRATE will tend to be most effective for sequences with a length of > 100 residues."
                "</span>", unsafe_allow_html=True)
    residues_to_modify = st.text_input(label="If you want to modify only specific residues, enter them here as "
                                             "comma-separated integers. If you add residues to this list, then any "
                                             "residues not in the list will be ignored by the AI. If you leave this "
                                             "blank, then the AI will assume all residues may be modified other than "
                                             "any residues you previously indicated you don't want modified.",
                                       value=None)

    gamma = int(st.number_input(label="Number of mutations to test for each site?", min_value=3))
    epsilon = int(st.number_input(label="Number of sites to test each round of mutation?", min_value=3))
    selection_criterion = st.selectbox(label="How to score permutations?", options=("plddt", "disorder"))

    with st.form(key="my_form_to_submit"):
        submit_button = st.form_submit_button(label="submit")

    if submit_button:
        if 22 > len(sequence) > 2048:

            st.write("Please input a sequence with a valid number of residues.")
            return

        out_df = pirate.PirateEvolutionTool(sequence, max_residues_to_alter, max_iterations, str(protected_residues),
                                            str(residues_to_modify), gamma, epsilon,
                                            selection_criterion).mutation_tool()

        if len(out_file_name) < 1:
            out_file_name = "pirate_sequences.csv"

        if out_file_name[-4:].lower() != ".csv":
            out_file_name += ".csv"

        st.download_button(label="Download PIRATE generated sequences",
                           data=out_df.to_csv(index=False).encode('utf-8'),
                           file_name=out_file_name,
                           mime="text/csv")
