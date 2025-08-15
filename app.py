import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import norm
import scipy.special

# Import custom modules
from src.data_preprocessing import preprocess_iEEG, artifact_rejection, parse_eye_tracking_traces, map_video_frames_to_eye_tracking, data_fusion, FaceDynamicsModel, RestingFaceEstimator
from src.models import MTPA, IdentityClassifier, SparseCCA
from src.simulators import simulate_iEEG_data, simulate_face_data, simulate_confusion_matrices, simulate_behavioral_psychophysics_data
from src.utils import plot_time_course, plot_confusion_matrix, plot_variance_explained, plot_posterior_probabilities, plot_scatter_with_regression, run_permutation_test, calculate_pairwise_accuracy, calculate_correlation_accuracy

st.set_page_config(layout="wide", page_title="Neurodynamic Basis of Real-World Face Perception")

# --- Helper Functions for Simulation and Caching ---
@st.cache_data
def get_simulated_data(seed=42):
    np.random.seed(seed)
    
    # Chapter 2 Data
    num_trials_ch2 = 200
    num_timepoints_ch2 = 500 # 500ms
    num_electrodes_ch2 = 75
    
    frp_ch2, bha_ch2, time_points_ch2 = simulate_iEEG_data(num_trials_ch2, num_timepoints_ch2, num_electrodes_ch2)
    # Combine FRP and BHA for MTPA features (flatten time and electrodes)
    # For MTPA, features are (N_trials, N_time_window_samples * N_electrodes)
    # Let's simplify for the demo: use a fixed time point's data or average over a window
    # For MTPA, we need (N_trials, N_timepoints, N_electrodes)
    neural_data_ch2 = np.concatenate((frp_ch2, bha_ch2[:, ::10, :]), axis=1) # Simulate 100Hz BHA, concatenate along time axis
    # Reshape for MTPA input (N_trials, N_timepoints_combined, N_electrodes)
    # This is a simplification. Actual MTPA combines ERP and ERBB in a window.
    # Let's just use FRP for MTPA demo for simplicity of feature extraction.
    
    # Simulate labels for viewpoint classification (5 categories)
    viewpoint_labels_ch2 = np.random.randint(0, 5, num_trials_ch2)
    
    # Simulate confusion matrices for mixture model
    conf_matrices_ch2 = simulate_confusion_matrices(num_samples=100, num_categories=5)
    
    # Simulate identity data for identity classification
    num_trials_identity = 100 # Subset for identity
    neural_data_identity = np.random.rand(num_trials_identity, 330 * 47) # Simplified features
    identity_labels_ch2 = np.random.choice(['ID_A', 'ID_B', 'ID_C', 'ID_D'], num_trials_identity)
    viewpoint_labels_identity = np.random.choice(['Left Away', 'Right Tilt', 'Straight', 'Right Away', 'Left Tilt'], num_trials_identity)
    
    # Chapter 4 Data
    num_fixations_ch4 = 500
    face_params_dim_ch4 = 224 # For expression analysis (identity removed)
    
    face_trajectories_ch4, fixation_identities_ch4 = simulate_face_data(num_fixations_ch4, face_params_dim=face_params_dim_ch4)
    # Simulate corresponding brain activity for CCA
    num_electrodes_ch4 = 150 # Example for Chapter 4
    iEEG_frp_ch4, iEEG_bha_ch4, _ = simulate_iEEG_data(num_fixations_ch4, 300, num_electrodes_ch4) # 300ms fixation
    # Flatten iEEG data for CCA (N_fixations, Q_brain_features)
    # Q = E * (300 + 30) = 150 * 330 = 49500
    brain_activity_ch4 = np.concatenate((iEEG_frp_ch4.reshape(num_fixations_ch4, -1), 
                                         iEEG_bha_ch4.reshape(num_fixations_ch4, -1)), axis=1)
    
    # Simulate resting face for expression analysis
    resting_face_params_ch4 = np.random.rand(face_params_dim_ch4) * 0.05 # Small vector
    
    # Behavioral psychophysics data
    psychophysics_data = simulate_behavioral_psychophysics_data()

    return {
        "frp_ch2": frp_ch2, "bha_ch2": bha_ch2, "time_points_ch2": time_points_ch2,
        "viewpoint_labels_ch2": viewpoint_labels_ch2, "conf_matrices_ch2": conf_matrices_ch2,
        "neural_data_identity": neural_data_identity, "identity_labels_ch2": identity_labels_ch2,
        "viewpoint_labels_identity": viewpoint_labels_identity,
        "face_trajectories_ch4": face_trajectories_ch4, "brain_activity_ch4": brain_activity_ch4,
        "resting_face_params_ch4": resting_face_params_ch4,
        "psychophysics_data": psychophysics_data,
        "fixation_identities_ch4": fixation_identities_ch4
    }

# --- Streamlit App Layout ---

st.title("🧠 The Neurodynamic Basis of Real-World Face Perception")
st.markdown("---")

# --- 1. Research Problem & Summaries ---
st.header("1. Research Overview")

st.subheader("Research Problem")
st.markdown("""
Understanding how our brains process information, specifically face perception, during natural, unconstrained real-world social interactions.
Most neuroscientific discoveries come from controlled lab experiments, raising questions about their ecological validity. Real-world studies are crucial to validate lab findings, understand generalization, and gain new insights not observable in artificial settings. This thesis aims to address the engineering, analytical, and ethical challenges of studying the brain in the real world, focusing on face perception.
""")

st.subheader("High-Level Summary (For General Audience)")
st.markdown("""
Our brains are amazing at recognizing faces and understanding emotions, but most of what we know comes from studies where people look at pictures on a screen in a lab. This research explores how our brains handle faces in the real world, like when you're talking to friends or family. We used special eye-tracking glasses and brain recordings from epilepsy patients in the hospital, capturing hours of their natural interactions.

We developed new computer tools to analyze these real-world videos and brain signals. We found that we could actually recreate videos of the faces people were looking at just from their brain activity! This shows our methods are powerful enough to study the messy, unpredictable real world. We also discovered something new about how our brains process facial expressions: they seem to be more sensitive to *what kind* of expression someone is making (e.g., a happy smile vs. a sympathetic smile) than to *how intense* that expression is. Plus, our brains are better at noticing small changes in a neutral face than small changes in a very expressive face, similar to how we notice a 1-pound difference in light weights but not in heavy ones. This suggests our brains use a "resting face" as a reference point for understanding expressions. This work helps us understand how our brains work in everyday life, not just in a lab.
""")

st.subheader("Detailed Technical Summary (For ML Experts)")
st.markdown("""
This thesis investigates the neurodynamic basis of real-world face perception, addressing the ecological validity gap in traditional neuroscience. Chapter 2 establishes foundational insights from controlled iEEG experiments on face viewpoint and identity representations in human ventral temporal cortex (VTC). Using Multivariate Temporal Pattern Analysis (MTPA) with LDA classifiers, it demonstrates robust decoding of face viewpoint (peak accuracy at 220ms) and identity (viewpoint-dependent, mirror-invariant, viewpoint-invariant). A novel Confusion Matrix Mixture Model, employing an EM algorithm and 1-SE rule for model selection, revealed a dominant "null" component and "anchoring" effects (strong coding for extreme viewpoints, weak for intermediate) in both data-driven (4 components, 61% variance explained) and hypothesis-driven (3 components, 40% variance explained) analyses. Critically, it identified a previously unreported "weaker mirror confusion" in mirror-symmetric representations. RSA with the EIG network showed early peaks for both linear and mirror-symmetric layers, with the latter exhibiting higher correlation. Identity decoding revealed a hierarchical progression, and a strong correlation between the mirror-symmetric viewpoint representation and the identity code, suggesting that purely feedforward propagation is insufficient to explain VTC dynamics.

Chapter 3 details a novel paradigm for collecting multimodal (iEEG, mobile eye-tracking, egocentric video, audio) data during unscripted social interactions in an Epilepsy Monitoring Unit (EMU). It outlines comprehensive data acquisition, ergonomic modifications for eye-tracking glasses, and robust preprocessing pipelines for eye-tracking (manual gaze correction), video (YOLO v3, OpenFace, human annotation), and audio (deep learning speech detection/diarization, manual correction). A precise data fusion strategy, anchored to eye-tracking events and utilizing digital triggers, addresses sampling rate variability and data corruption, ensuring high-quality multimodal datasets.

Chapter 4 leverages this real-world paradigm to reconstruct the neural code for dynamic facial expressions. Faces in egocentric video were parameterized using Deep 3D Face and eye-gaze models, and their dynamics modeled with a linear state-space model. Sparse Canonical Correlation Analysis (CCA) was employed to learn a "neuro-perceptual space" jointly from face trajectories and fixation-locked brain activity (FRP and FRBHA). The model demonstrated robust bidirectional reconstruction: qualitatively accurate face videos from brain activity and significant brain activity reconstruction (p < 0.05) across temporal-parietal junction (social-vision pathway) and VTC. Probing the geometry of this neuro-perceptual space, after recentering expressions around an individual's "resting face" (norm-based coding), revealed two key findings: 1) Neural population tuning was sharper for differences in expression *type* (tangential distances) than *intensity* (radial distances). 2) Neural sensitivity (prediction error) decreased as expression intensity increased, demonstrating an analog of Weber's law for facial expressions. These findings, validated by a psychophysical experiment, suggest an oval-shaped neural tuning for facial expressions, oriented towards the resting norm, with increasing tuning width further from the norm. The thesis concludes by emphasizing the framework's potential for ecologically valid neuroscientific discovery and future directions in expanding modalities and scaling real-world neuroscience.
""")

st.markdown("---")

# --- 2. Methodology & Interactive Demos ---
st.header("2. Methodology & Interactive Demos")

st.subheader("2.1. Chapter 2: The Neural Code for Face Viewpoint and Identity (Controlled Experiments)")
st.markdown("""
This section explores findings from controlled laboratory experiments using intracranial EEG (iEEG) to understand how the brain processes face viewpoint and identity.
""")

with st.expander("Experimental Setup & Data (Chapter 2)"):
    st.markdown("""
    **Participants:** 18 human subjects (11 males, 7 females) with iEEG implants for epilepsy monitoring.
    **Experiments:**
    *   **Functional Localizer:** One-back task with various object categories (faces, bodies, words, etc.).
    *   **Face Perception (Gender Discrimination):** Participants viewed faces from the Karolinska Directed Emotional Faces (KDEF) database with 5 distinct viewpoints (e.g., Left Away, Right Tilt, Straight). Three stimulus variants were used.
    **Data Acquisition:** iEEG recordings from 75 face-selective electrodes at 1 KHz.
    **Preprocessing:** iEEG signals processed for Response Potentials (FRP) and Broadband High Frequency Activity (BHA). Face-selective electrodes identified based on anatomical constraints, signal strength, and statistical sensitivity.
    """)

st.subheader("2.1.1. Face Viewpoint Classification (MTPA)")
st.markdown("""
Multivariate Temporal Pattern Analysis (MTPA) was used to decode face viewpoint from neural activity.
""")
if st.button("Run MTPA Simulation"):
    data = get_simulated_data()
    frp_ch2 = data["frp_ch2"]
    bha_ch2 = data["bha_ch2"]
    time_points_ch2 = data["time_points_ch2"]
    viewpoint_labels_ch2 = data["viewpoint_labels_ch2"]
    
    # For MTPA, we need to combine FRP and BHA into features per trial per time window.
    # Let's simplify for the demo: use FRP as the primary feature set for MTPA.
    # The paper states "both ERP and ERBB signals in the time window are combined as input features".
    # For simulation, we'll just use FRP for simplicity of feature vector creation.
    # In a real scenario, you'd concatenate or combine the time-windowed FRP and BHA.
    
    # Simulate features for MTPA: (N_trials, N_timepoints, N_electrodes)
    # For the demo, let's use a simplified feature set for MTPA.
    # The MTPA class expects (N_trials, N_timepoints, N_electrodes)
    
    # Let's create a dummy combined neural data for MTPA
    # Assuming FRP is (N_trials, N_timepoints, N_electrodes)
    # And BHA is (N_trials, N_timepoints/10, N_electrodes)
    # We need to align them. For simplicity, let's just use FRP for the demo.
    
    # To make it more realistic, let's combine them by concatenating features for a given time window.
    # The MTPA class expects (N_trials, N_timepoints, N_electrodes)
    # Let's assume `frp_ch2` is the input to MTPA, and it will handle windowing.
    
    mtpa = MTPA(classifier=LinearDiscriminantAnalysis(), time_window_ms=100)
    
    # Run MTPA. The `run` method expects (N_trials, N_timepoints, N_electrodes)
    # Let's use a subset of electrodes for faster simulation
    num_electrodes_for_mtpa = 20
    sim_neural_data_for_mtpa = frp_ch2[:, :, :num_electrodes_for_mtpa] # Use FRP for simplicity
    
    with st.spinner("Running MTPA simulation..."):
        accuracy_history, d_prime_history, conf_matrices_history = mtpa.run(
            sim_neural_data_for_mtpa, viewpoint_labels_ch2, time_points_ch2
        )
    
    st.success("MTPA Simulation Complete!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_time_course(
            np.array([accuracy_history]), time_points_ch2[:len(accuracy_history)],
            "Population Averaged Classification Accuracy vs. Time", "Accuracy", chance_level=0.2
        ))
    with col2:
        st.pyplot(plot_time_course(
            np.array([d_prime_history]), time_points_ch2[:len(d_prime_history)],
            "Population Averaged d' vs. Time", "d'"
        ))
    
    st.markdown("---")
    st.subheader("Average Confusion Matrix at Peak Accuracy")
    peak_idx = np.argmax(accuracy_history)
    avg_cm_at_peak = conf_matrices_history[peak_idx]
    viewpoint_categories = ["Left Away", "Right Tilt", "Straight", "Right Away", "Left Tilt"]
    st.pyplot(plot_confusion_matrix(avg_cm_at_peak, viewpoint_categories, f"Average Confusion Matrix at {time_points_ch2[peak_idx]:.0f} ms"))

st.subheader("2.1.2. Confusion Matrix Mixture Model")
st.markdown("""
A novel mixture model approach was developed for data-driven representational analysis of face viewpoint.
""")

model_type = st.radio("Choose Mixture Model Type:", ("Data-Driven", "Hypothesis-Driven"))

if st.button("Run Mixture Model Simulation"):
    data = get_simulated_data()
    conf_matrices_ch2 = data["conf_matrices_ch2"]
    
    if model_type == "Data-Driven":
        st.info("Simulating data-driven mixture model fitting. Optimal K=4 is expected based on paper.")
        # For demo, we'll just fit a K=4 model directly and show its components.
        # Model selection (cross-validation, 1-SE rule) is computationally intensive for a demo.
        K_optimal = 4
        model = ConfusionMatrixMixtureModel(K=K_optimal, D=5)
        with st.spinner(f"Fitting {K_optimal}-component data-driven mixture model..."):
            model.fit(conf_matrices_ch2)
        st.success("Data-Driven Mixture Model Simulation Complete!")
        
        components, priors = model.get_components()
        st.write(f"**Learned Components (K={K_optimal}):**")
        for k in range(K_optimal):
            st.write(f"**Component {k+1} (Prior: {priors[k]:.4f})**")
            st.dataframe(pd.DataFrame(components[k], columns=viewpoint_categories, index=viewpoint_categories).round(3))
            
        st.markdown("---")
        st.subheader("Simulated Variance Explained")
        # Simulate variance explained based on paper's findings (61% for K=4)
        variance_explained_data = [0, 30, 45, 61, 65, 68, 70, 71, 72, 73] # Example progression
        st.pyplot(plot_variance_explained(variance_explained_data[:K_optimal], "Simulated Variance Explained by Data-Driven Mixture Model"))
        
        st.markdown("---")
        st.subheader("Simulated Posterior Probabilities Over Time")
        # Simulate posterior probabilities over time
        num_time_points_sim = 50 # Simplified time points for posteriors
        sim_posteriors = np.random.rand(num_time_points_sim, K_optimal)
        sim_posteriors = sim_posteriors / np.sum(sim_posteriors, axis=1, keepdims=True) # Normalize
        sim_time_points = np.linspace(0, 500, num_time_points_sim) # 0-500ms
        
        component_labels = [f'C{i+1}' for i in range(K_optimal)]
        st.pyplot(plot_posterior_probabilities(sim_posteriors, sim_time_points, component_labels, "Simulated Posterior Probabilities Over Time"))

    else: # Hypothesis-Driven
        st.info("Simulating hypothesis-driven mixture model fitting. Optimal K=3 is expected based on paper.")
        K_optimal = 3
        # Define simplified structured templates
        null_template = get_structured_template('Null')
        linear_relaxed_template = get_structured_template('Linear Angle') # Simplified as 'Linear Angle'
        mirror_symmetric_relaxed_template = get_structured_template('Mirror Symmetric') # Simplified as 'Mirror Symmetric'
        
        # For demo, we'll just use these fixed templates as components.
        # In a real scenario, their parameters would be estimated.
        
        # Create a dummy model with these templates as components
        # This is a conceptual representation, not a true fit.
        components_hd = np.array([null_template, linear_relaxed_template, mirror_symmetric_relaxed_template])
        priors_hd = np.array([0.6, 0.3, 0.1]) # Example priors
        priors_hd = priors_hd / np.sum(priors_hd)
        
        st.success("Hypothesis-Driven Mixture Model Simulation Complete (Conceptual)!")
        
        st.write(f"**Hypothesized Components (K={K_optimal}):**")
        template_names = ["Null", "Linear Relaxed", "Mirror Symmetric IV"]
        for k in range(K_optimal):
            st.write(f"**Component {template_names[k]} (Prior: {priors_hd[k]:.4f})**")
            st.dataframe(pd.DataFrame(components_hd[k], columns=viewpoint_categories, index=viewpoint_categories).round(3))
            
        st.markdown("---")
        st.subheader("Simulated Variance Explained")
        variance_explained_data_hd = [0, 33, 40, 41, 41.5, 41.8, 41.9] # Example progression
        st.pyplot(plot_variance_explained(variance_explained_data_hd[:K_optimal], "Simulated Variance Explained by Hypothesis-Driven Mixture Model"))
        
        st.markdown("---")
        st.subheader("Simulated Posterior Probabilities Over Time")
        num_time_points_sim = 50
        sim_posteriors_hd = np.random.rand(num_time_points_sim, K_optimal)
        sim_posteriors_hd = sim_posteriors_hd / np.sum(sim_posteriors_hd, axis=1, keepdims=True)
        sim_time_points = np.linspace(0, 500, num_time_points_sim)
        
        st.pyplot(plot_posterior_probabilities(sim_posteriors_hd, sim_time_points, template_names, "Simulated Posterior Probabilities Over Time (Hypothesis-Driven)"))

st.subheader("2.1.3. Identity Decoding")
st.markdown("""
Identity decoding was performed for viewpoint-dependent, mirror-invariant, and viewpoint-invariant contexts.
""")
if st.button("Run Identity Decoding Simulation"):
    data = get_simulated_data()
    neural_data_identity = data["neural_data_identity"]
    identity_labels_ch2 = data["identity_labels_ch2"]
    viewpoint_labels_identity = data["viewpoint_labels_identity"]
    
    identity_classifier = IdentityClassifier()
    
    st.info("Simulating identity decoding for different contexts...")
    
    vp_dep_acc = identity_classifier.classify(neural_data_identity, identity_labels_ch2, viewpoint_labels_identity, 'viewpoint_dependent')
    mirror_inv_acc = identity_classifier.classify(neural_data_identity, identity_labels_ch2, viewpoint_labels_identity, 'mirror_invariant')
    vp_inv_acc = identity_classifier.classify(neural_data_identity, identity_labels_ch2, viewpoint_labels_identity, 'viewpoint_invariant')
    
    st.success("Identity Decoding Simulation Complete!")
    
    st.write(f"**Viewpoint Dependent Identity Accuracy:** {vp_dep_acc:.2f}")
    st.write(f"**Mirror Invariant Identity Accuracy:** {mirror_inv_acc:.2f}")
    st.write(f"**Viewpoint Invariant Identity Accuracy:** {vp_inv_acc:.2f}")
    st.markdown("""
    *Note: Simulated accuracies are illustrative and may not perfectly match paper's exact values due to simplified data generation.*
    """)

st.markdown("---")

st.subheader("2.2. Chapter 3: A New Paradigm for Investigating Real World Social Behavior")
st.markdown("""
This chapter describes the methodology for collecting and processing multimodal data (iEEG, eye-tracking, video, audio) during unscripted social interactions in a hospital setting.
""")

with st.expander("Data Acquisition & Preprocessing (Chapter 3)"):
    st.markdown("""
    **Participants:** 6 epilepsy patients in an Epilepsy Monitoring Unit (EMU).
    **Behavioral Data:** SensoMotoric Instrument’s (SMI) ETG 2 Eye Tracking Glasses captured egocentric video (1280x960, 24fps), eye-tracking (60Hz), and audio (16KHz).
    **Physiological Data:** Intracranial EEG (iEEG) from 96-220 electrodes at 1KHz using Ripple Neuro's Grapevine NIP.
    **Synchronization:** Digital triggers broadcasted every 10 seconds to align iEEG and eye-tracking streams.
    **Preprocessing Steps:**
    *   **Eye-Tracking:** Parsing into fixation, saccade, blink events; manual correction for gaze errors.
    *   **Video:** Automated object/face detection (YOLO v3), human annotation for verification, OpenFace for facial landmarks/pose/expressions.
    *   **Audio:** Deep learning for speech detection and speaker diarization, manual verification.
    *   **iEEG:** Filtering (FRP, BHA), artifact rejection (amplitude, std dev, consecutive change).
    **Data Fusion:** Eye-tracking as reference; mapping video frames (with frame rate variability correction) and audio segments to eye-tracking events; aligning iEEG data to eye-tracking events.
    """)

st.subheader("2.2.1. Simulated Behavioral Data Summary")
st.markdown("""
Real-world behavior is highly heterogeneous. Here's a simulated summary of behavioral data collected.
""")
if st.button("Simulate Behavioral Data Summary"):
    st.info("Generating simulated behavioral data statistics...")
    
    # Simulate data for 3 recording sessions
    session_data = []
    for i in range(3):
        total_duration = np.random.uniform(60, 120) # minutes
        saccade_frac = np.random.uniform(0.10, 0.15)
        fixation_frac = np.random.uniform(0.70, 0.80)
        blink_frac = 1 - saccade_frac - fixation_frac
        
        on_face_frac = np.random.uniform(0.20, 0.40) # % time fixating on faces
        
        speech_frac = np.random.uniform(0.30, 0.60)
        participant_speech_frac = np.random.uniform(0.4, 0.6) * speech_frac
        other_speech_frac = speech_frac - participant_speech_frac
        
        session_data.append({
            'Session': f'S{i+1}',
            'Total Duration (min)': total_duration,
            'Saccade (%)': saccade_frac * 100,
            'Fixation (%)': fixation_frac * 100,
            'Blink (%)': blink_frac * 100,
            'Fixation on Faces (%)': on_face_frac * 100,
            'Speech (%)': speech_frac * 100,
            'Participant Speech (%)': participant_speech_frac * 100,
            'Other Speech (%)': other_speech_frac * 100
        })
    
    df_behavior = pd.DataFrame(session_data)
    st.dataframe(df_behavior.round(2))
    
    st.markdown("""
    *   **Visual Behavior:** Saccades typically 10-15% of duration, fixations 70-80%. Fixations on faces are less than 30-40% of total fixation time, even in social situations, but face fixations tend to be slightly longer.
    *   **Auditory Context:** Varying levels of verbal discourse. Participant's speech is most reliably detected.
    """)

st.markdown("---")

st.subheader("2.3. Chapter 4: Reconstructing the Neural Code for Real World Face Perception")
st.markdown("""
This section applies the real-world paradigm to investigate face perception, focusing on reconstructing faces from brain activity and unraveling the neural code for facial expressions.
""")

with st.expander("Face Parameterization & Dynamics (Chapter 4)"):
    st.markdown("""
    **Face Parameterization:** Each detected face in egocentric video is represented in a linear face model (Deep 3D Face) for pose, shape, texture, and expression. Eye-gaze is also estimated. This results in a 229-dimensional representation per face per frame.
    **Face Dynamics Model:** A state space model tracks the dynamics of shape, texture, and expression in a low-dimensional latent space (R^30).
    **Resting Face Estimation:** The average pose-corrected shape, texture, and expression for each individual is estimated as their "resting face" (norm). This is subtracted from fixated faces to analyze expressions independent of identity.
    """)

st.subheader("2.3.1. Face & Brain Reconstruction (Sparse CCA)")
st.markdown("""
Sparse Canonical Correlation Analysis (CCA) is used to learn a "neuro-perceptual space" where brain activity and facial features are highly correlated, enabling bidirectional reconstruction.
""")

if st.button("Run Sparse CCA Simulation & Reconstruction"):
    data = get_simulated_data()
    face_trajectories_ch4 = data["face_trajectories_ch4"]
    brain_activity_ch4 = data["brain_activity_ch4"]
    
    st.info("Fitting Sparse CCA model...")
    # Use a smaller number of components for faster demo
    n_components_cca = 5
    cca_model = SparseCCA(n_components=n_components_cca)
    
    # For demo, we'll use a simplified fit. In reality, this involves cross-validation.
    # We need to ensure X and Y have enough samples for the dimensions.
    # Let's reduce the dimensions for simulation if they are too large.
    
    # Simulate a smaller feature space for demo if original is too large
    P_face_sim = 50 # Reduced face features for demo
    Q_brain_sim = 100 # Reduced brain features for demo
    
    face_trajectories_sim = face_trajectories_ch4[:, :P_face_sim]
    brain_activity_sim = brain_activity_ch4[:, :Q_brain_sim]
    
    with st.spinner("Fitting Sparse CCA... (This is a simplified simulation)"):
        cca_model.fit(face_trajectories_sim, brain_activity_sim)
    st.success("Sparse CCA Simulation Complete!")
    
    st.subheader("Reconstruction Results")
    
    # Simulate pairwise classification accuracy for face reconstruction
    # In a real scenario, this would use held-out data.
    Y_cc_sim, X_cc_sim = cca_model.transform(face_trajectories_sim, brain_activity_sim)
    
    # Ensure singular_values are not all zero for weighting
    singular_values_for_acc = cca_model.singular_values
    if np.sum(singular_values_for_acc) < 1e-9:
        singular_values_for_acc = np.ones_like(singular_values_for_acc) # Avoid division by zero
        
    face_reconstruction_accuracy = calculate_pairwise_accuracy(Y_cc_sim, X_cc_sim, singular_values_for_acc)
    st.write(f"**Simulated Face Reconstruction Accuracy (from Brain Activity):** {face_reconstruction_accuracy:.2f} (Chance: 0.50)")
    st.markdown("*(Paper reports significant accuracy for all patients)*")
    
    # Simulate brain activity reconstruction correlation
    reconstructed_brain_sim = cca_model.reconstruct_brain(X_cc_sim)
    brain_reconstruction_corr = calculate_correlation_accuracy(brain_activity_sim, reconstructed_brain_sim)
    st.write(f"**Simulated Brain Activity Reconstruction Correlation (from Faces):** {brain_reconstruction_corr:.2f}")
    st.markdown("*(Paper reports robust reconstructions across several cortical areas, especially temporal-parietal junction)*")
    
    st.subheader("Conceptual Neuro-Perceptual Space")
    st.markdown("""
    The neuro-perceptual space is jointly learned, where movement along its axes corresponds to parametric changes in both brain activity and facial features.
    Below is a conceptual representation.
    """)
    
    st.info("Conceptual Visualization: Imagine sliders that let you move along these axes, and you'd see the predicted face change (e.g., expression, pose) and the predicted brain activity pattern change simultaneously.")
    
    # Display conceptual axes (weights)
    W_brain, W_face, s_vals = cca_model.get_neuro_perceptual_axes()
    
    st.write(f"**Canonical Correlations (Singular Values):** {s_vals.round(3)}")
    
    st.write("**Brain Canonical Vectors (Conceptual Weights for Electrodes):**")
    st.dataframe(pd.DataFrame(W_brain[:, :2], columns=[f'Axis {i+1}' for i in range(2)]).head())
    st.write("*(Showing first 5 rows of first 2 axes)*")
    
    st.write("**Face Canonical Vectors (Conceptual Weights for Face Features):**")
    st.dataframe(pd.DataFrame(W_face[:, :2], columns=[f'Axis {i+1}' for i in range(2)]).head())
    st.write("*(Showing first 5 rows of first 2 axes)*")

st.subheader("2.3.2. Neural Population Tuning for Facial Expressions")
st.markdown("""
The study investigated how our brains code for variations in facial expressions, hypothesizing a norm-based code and differential sensitivity.
""")

if st.button("Analyze Neural Population Tuning Simulation"):
    data = get_simulated_data()
    face_trajectories_ch4 = data["face_trajectories_ch4"]
    brain_activity_ch4 = data["brain_activity_ch4"]
    resting_face_params_ch4 = data["resting_face_params_ch4"]
    
    # Simulate a simplified CCA model for tuning analysis
    n_components_tuning = 3
    cca_tuning_model = SparseCCA(n_components=n_components_tuning)
    
    # For tuning analysis, face_trajectories are assumed to be identity-removed and centered
    # Let's use the simulated face_trajectories_ch4 directly as the "deviations from resting face"
    # and brain_activity_ch4 as the corresponding neural data.
    
    P_face_tuning_sim = 30 # Reduced face features for tuning demo
    Q_brain_tuning_sim = 50 # Reduced brain features for tuning demo
    
    face_trajectories_tuning_sim = face_trajectories_ch4[:, :P_face_tuning_sim]
    brain_activity_tuning_sim = brain_activity_ch4[:, :Q_brain_tuning_sim]
    
    with st.spinner("Fitting Sparse CCA for tuning analysis..."):
        cca_tuning_model.fit(face_trajectories_tuning_sim, brain_activity_tuning_sim)
    
    Y_cc_tuning, X_cc_tuning = cca_tuning_model.transform(face_trajectories_tuning_sim, brain_activity_tuning_sim)
    
    st.success("Neural Population Tuning Analysis Simulation Complete!")
    
    st.subheader("Hypothesis 1: Sensitivity to Expression Type vs. Intensity")
    st.markdown("""
    *   **Hypothesis:** Brain is more sensitive to differences in the *kind* of expression (tangential distances) than the *intensity* of the expression (radial distances).
    *   **Finding:** Neural tuning was sharper (steeper slope) for differences in expression relative to differences in intensity.
    """)
    
    # Calculate radial and tangential distances from simulated face trajectories
    # For this demo, we'll assume face_trajectories_tuning_sim are already the 'r' vectors (deviations from norm)
    radial_dists, tangential_dists = calculate_radial_tangential_distances(face_trajectories_tuning_sim, resting_face_params_ch4[:P_face_tuning_sim])
    
    # Calculate neural distances
    neural_dists = calculate_neural_distances(Y_cc_tuning)
    
    # Ensure lengths match for plotting (some pairs might be skipped in distance calculation)
    min_len = min(len(radial_dists), len(tangential_dists), len(neural_dists))
    radial_dists = radial_dists[:min_len]
    tangential_dists = tangential_dists[:min_len]
    neural_dists = neural_dists[:min_len]
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_scatter_with_regression(
            radial_dists, neural_dists,
            "Neural Distance vs. Radial Face Distance", "Radial Face Distance", "Neural Distance"
        ))
    with col2:
        st.pyplot(plot_scatter_with_regression(
            tangential_dists, neural_dists,
            "Neural Distance vs. Tangential Face Distance", "Tangential Face Distance", "Neural Distance"
        ))
    st.info("*(Observe the slopes: a steeper slope for tangential distances would support the hypothesis.)*")
    
    st.subheader("Hypothesis 2: Weber's Law Analog for Facial Expressions")
    st.markdown("""
    *   **Hypothesis:** Brain is more sensitive to subtle differences from a person's resting expression than from strong expressions (i.e., sensitivity decreases as expression intensity increases).
    *   **Finding:** Neural sensitivity (error of neural prediction) increased with the intensity of expressions.
    """)
    
    # Calculate expression intensity (distance from resting face)
    expression_intensity = calculate_expression_intensity(face_trajectories_tuning_sim, resting_face_params_ch4[:P_face_tuning_sim])
    
    # Simulate neural prediction error (e.g., inverse of correlation or just random noise increasing with intensity)
    # For demo, let's use a simple increasing function
    neural_prediction_error = expression_intensity * 0.5 + np.random.rand(len(expression_intensity)) * 0.1
    
    st.pyplot(plot_scatter_with_regression(
        expression_intensity, neural_prediction_error,
        "Neural Prediction Error vs. Expression Intensity", "Expression Intensity (Distance from Resting Face)", "Neural Prediction Error"
    ))
    st.info("*(Observe the trend: an increasing trend in error would support the hypothesis.)*")

st.subheader("2.3.3. Behavioral Experiment Validation")
st.markdown("""
A psychophysics experiment was conducted to validate the neural findings behaviorally.
""")
if st.button("Run Behavioral Psychophysics Simulation"):
    psychophysics_data = simulate_behavioral_psychophysics_data()
    
    st.success("Behavioral Psychophysics Simulation Complete!")
    
    st.subheader("Accuracy for Expression Type vs. Intensity")
    radial_acc = psychophysics_data[psychophysics_data['type'] == 'radial']['accuracy'].mean()
    tangential_acc = psychophysics_data[psychophysics_data['type'] == 'tangential']['accuracy'].mean()
    
    st.write(f"**Average Accuracy for Radial Perturbations (Intensity Change):** {radial_acc:.2f}")
    st.write(f"**Average Accuracy for Tangential Perturbations (Type Change):** {tangential_acc:.2f}")
    st.info("*(Paper reports greater accuracy for discrimination between expression type vs. intensity, consistent with neural findings.)*")
    
    st.subheader("Accuracy vs. Expression Intensity (Weber's Law Analog)")
    # Group by intensity bins for plotting
    psychophysics_data['intensity_bin'] = pd.cut(psychophysics_data['intensity'], bins=5, labels=False)
    avg_acc_by_intensity = psychophysics_data.groupby('intensity_bin')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='intensity_bin', y='accuracy', data=avg_acc_by_intensity, marker='o')
    plt.title("Behavioral Accuracy vs. Expression Intensity")
    plt.xlabel("Expression Intensity Bin (Low to High)")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    st.pyplot(plt)
    st.info("*(Paper reports decreasing sensitivity/accuracy for facial expressions as expression intensity increased, consistent with neural findings.)*")

st.markdown("---")

# --- 3. Limitations ---
st.header("3. Limitations")
st.markdown("""
The research acknowledges several limitations:

*   **Ecological Validity vs. Control:** Inherent trade-off between real-world complexity and experimental control.
*   **iEEG Specifics:**
    *   **Sampling Bias:** Imbalance in hemispheric electrode sampling due to clinical necessity.
    *   **Patient Population:** Findings from epilepsy patients may not fully generalize to healthy populations.
    *   **Limited Mobility:** Hospital room setting, not fully free-roaming.
*   **Data Acquisition Challenges:** Potential for eye-tracking errors, variable audio quality, and video frame issues requiring extensive manual correction.
*   **Annotation Challenges:** Automated annotation requires significant human verification due to accuracy limitations.
*   **Model Specifics:**
    *   **Mixture Model:** EM algorithm convergence to local minima, label switching problem in bootstrapping.
    *   **Sparse CCA:** Linearity might limit capturing complex non-linear relationships. Norm-based coding supported but not directly competed against other schemes. Sample complexity can be an issue for small datasets.
*   **Ethical Considerations:** Complex issues of patient and visitor privacy, data sharing, and obtaining ongoing informed consent in a clinical environment.
""")

st.markdown("---")

# --- 4. Suggested Improvements ---
st.header("4. Suggested Improvements")

st.subheader("4.1. Model Performance Improvements")
st.markdown("""
1.  **Advanced Deep Learning for Face Parameterization:** Explore more sophisticated and robust 3D face reconstruction models (e.g., SMPL-X, FLAME) that can capture subtle facial movements, expressions, and even gaze more accurately in unconstrained real-world videos, potentially integrating with optical flow for motion. This could provide richer, more granular input features for the CCA model.
2.  **Non-linear CCA Variants & Deep Learning:** While Sparse CCA is interpretable, its linearity might limit its ability to capture complex non-linear brain-behavior relationships. Investigate kernel CCA or deep CCA (e.g., Deep Canonical Correlation Analysis) to learn non-linear mappings between neural activity and face features. This could potentially improve reconstruction accuracy and reveal more intricate tuning properties, though interpretability would need careful consideration (e.g., using saliency maps or feature attribution methods on the deep networks).
3.  **Temporal Dynamics in Joint Space Learning:** The current Sparse CCA flattens time series into single vectors. Incorporate recurrent neural networks (RNNs) or temporal convolutional networks (TCNs) within a deep CCA framework to explicitly model the temporal dependencies and dynamics of both neural activity and facial movements, allowing the joint space to capture dynamic correlations more effectively. This would move beyond static "snapshots" of fixations.
""")

st.subheader("4.2. Paper Quality Improvements")
st.markdown("""
1.  **Quantitative Evaluation of Qualitative Reconstructions:** While "qualitatively accurate" reconstructions are shown, a more rigorous quantitative metric for video reconstruction quality (e.g., perceptual similarity metrics like SSIM, FID, or even human perceptual ratings) would strengthen the claims about the robustness of the approach. This would provide a more objective measure beyond pairwise classification accuracy.
2.  **Detailed Hyperparameter Tuning and Sensitivity Analysis:** For both the Mixture Model and Sparse CCA, provide more detailed information on the hyperparameter search space, the specific criteria used for selection (beyond the 1-SE rule), and a sensitivity analysis to show how robust the findings are to variations in these parameters. This enhances reproducibility and confidence in the results.
3.  **Direct Comparison of Competing Hypotheses:** For the neural code of facial expressions (norm-based vs. axis-based), explicitly define and train models for competing hypotheses and quantitatively compare their fit to the data (e.g., using model evidence or cross-validation performance). This would provide stronger evidence for the proposed oval-shaped tuning rather than just showing its emergence.
""")

st.markdown("---")
st.footer("Developed by BLACKBOX.AI Assistant based on 'The Neurodynamic Basis of Real World Face Perception' by Arish Alreja.")

