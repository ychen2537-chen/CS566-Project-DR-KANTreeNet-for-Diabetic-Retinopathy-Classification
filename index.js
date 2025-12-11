var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI } from "@google/genai";

const PROJECT_DETAILS = {
    title: "DR-KANTreeNet",
    subtitle: "Diabetic Retinopathy Classification via Kolmogorov-Arnold Tree Networks",
    abstract: "Diabetic Retinopathy (DR) is a leading cause of blindness. Traditional CNNs suffer from a lack of interpretability and struggle with the fine-grained feature discrimination required for early DR stages. This project proposes DR-KANTreeNet, a hybrid architecture combining Kolmogorov-Arnold Networks (KANs) for adaptive feature extraction with a Differentiable Neural Decision Tree (DNDT) head for transparent classification.",
    motivation: [
        { title: "Black Box Paradox", content: "Deep Learning models achieve high accuracy but lack transparency. In healthcare, a 'trust me' prediction is insufficient; clinicians need the 'why' behind a diagnosis.", icon: "fa-box-open" },
        { title: "Feature Nuance", content: "Early DR signs like microaneurysms are minute, pixel-level anomalies. Standard CNNs often lose these fine-grained details during pooling operations.", icon: "fa-microscope" },
        { title: "The KAN Advantage", content: "Kolmogorov-Arnold Networks learn activation functions on edges, offering superior function approximation for complex biological patterns with fewer parameters than MLPs.", icon: "fa-network-wired" }
    ],
    stack: ["PyTorch", "Python 3.9", "OpenCV", "Scikit-Learn", "APTOS 2019 Dataset"],
    results: {
        accuracy: "86.18%",
        auc: "0.9714",
        f1: "0.8620",
        kappa: "0.9116",
        comparison: [
            { model: "DL (Rakhlin, 2017)", auc: "-", acc: "82.5", f1: "80.3", kappa: "89.5" },
            { model: "CANet (Li et al., 2019a)", auc: "-", acc: "83.2", f1: "81.3", kappa: "90.0" },
            { model: "DANIL (Gong et al., 2020)", auc: "-", acc: "83.8", f1: "67.2", kappa: "-" },
            { model: "ResNet34 (He et al., 2016)", auc: "96.0", acc: "82.4", f1: "81.5", kappa: "88.1" },
            { model: "GREEN-SE-ResNet50 (Liu, 2020)", auc: "-", acc: "85.7", f1: "85.2", kappa: "91.2" },
            { model: "ViT (Dosovitskiy et al., 2020)", auc: "95.8", acc: "81.5", f1: "80.5", kappa: "88.3" },
            { model: "Swin (Liu et al., 2021)", auc: "96.0", acc: "83.7", f1: "83.0", kappa: "90.6" },
            { model: "TransMIL (Shao et al., 2021)", auc: "97.4", acc: "82.5", f1: "82.9", kappa: "91.5" },
            { model: "MIL-ViT (Bi et al., 2023)", auc: "97.1", acc: "83.2", f1: "84.0", kappa: "91.1" },
            { model: "MIL-ViTa", auc: "98.1", acc: "85.8", f1: "85.5", kappa: "92.3" },
            { model: "Graph-DR (Akter et al., 2025)", auc: "-", acc: "96.0", f1: "96.0", kappa: "95.0" },
            { model: "DR-KANTreeNet (Ours)", auc: "97.1", acc: "86.2", f1: "86.2", kappa: "91.2", highlight: true, bestMetrics: ["acc", "f1"] }
        ]
    }
};

const SYSTEM_INSTRUCTION = `You are the AI Research Assistant for the project "DR-KANTreeNet". 
Your goal is to explain the project to visitors.
Here are the project details:
${JSON.stringify(PROJECT_DETAILS)}

Key Technical Points:
1. Replaces standard MLP layers with KAN layers (learnable activations on edges).
2. Uses a Neural Decision Tree at the end for class routing, making decisions traceable.
3. Dataset: APTOS 2019 (Blindness Detection).
4. Preprocessing: CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian Filtering.

Answer questions concisely, professionally, and enthusiastically. If asked about code, describe the logical structure.`;
// --- Components ---
const Navigation = () => (_jsx("nav", { className: "fixed top-0 w-full z-50 glass-panel border-b border-slate-700", children: _jsx("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: _jsxs("div", { className: "flex items-center justify-between h-16", children: [_jsx("div", { className: "flex items-center", children: _jsxs("span", { className: "text-xl font-bold tracking-tight text-white", children: ["DR-", _jsx("span", { className: "text-cyan-400", children: "KANTreeNet" })] }) }), _jsx("div", { className: "hidden md:block", children: _jsx("div", { className: "ml-10 flex items-baseline space-x-4", children: ['Motivation', 'Approach', 'Implementation', 'Results', 'Discussion'].map((item) => (_jsx("a", { href: `#${item.toLowerCase()}`, className: "text-slate-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors", children: item }, item))) }) })] }) }) }));
const Hero = () => (_jsxs("header", { className: "relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden", children: [_jsxs("div", { className: "relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center z-10", children: [_jsxs("div", { className: "inline-flex items-center px-3 py-1 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-xs font-medium mb-6", children: [_jsx("i", { className: "fas fa-code-branch mr-2" }), " v1.0 Release"] }), _jsxs("h1", { className: "text-5xl md:text-7xl font-extrabold tracking-tight text-white mb-6", children: ["Interpretable Vision for ", _jsx("br", {}), _jsx("span", { className: "gradient-text", children: "Diabetic Retinopathy" })] }), _jsx("p", { className: "mt-4 max-w-2xl mx-auto text-xl text-slate-400", children: PROJECT_DETAILS.subtitle }), _jsxs("div", { className: "mt-8 flex justify-center gap-4", children: [_jsx("a", { href: "#results", className: "px-8 py-3 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold transition-all shadow-[0_0_20px_rgba(8,145,178,0.5)]", children: "View Results" }), _jsx("a", { href: "#approach", className: "px-8 py-3 rounded-lg glass-panel hover:bg-slate-800 text-white font-semibold transition-all", children: "How it Works" })] })] }), _jsxs("div", { className: "absolute top-0 left-0 w-full h-full overflow-hidden -z-10", children: [_jsx("div", { className: "absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" }), _jsx("div", { className: "absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl" })] })] }));
const Motivation = () => (_jsx("section", { id: "motivation", className: "py-24 bg-slate-900", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: [_jsxs("div", { className: "text-center mb-20", children: [_jsx("h2", { className: "text-3xl md:text-4xl font-bold text-white mb-4", children: "The Imperative for Innovation" }), _jsx("div", { className: "w-20 h-1 bg-cyan-500 mx-auto rounded-full" }), _jsx("p", { className: "mt-6 text-slate-400 max-w-2xl mx-auto text-lg", children: "Bridging the gap between silent pathology and automated detection through Computer Vision." })] }), _jsx("div", { className: "bg-slate-800/50 rounded-3xl border border-slate-700 overflow-hidden mb-20", children: _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2", children: [_jsxs("div", { className: "p-8 md:p-12 flex flex-col justify-center", children: [_jsxs("div", { className: "inline-flex items-center gap-2 text-red-400 font-bold tracking-wider text-xs uppercase mb-4", children: [_jsx("i", { className: "fas fa-search" }), " Visual Biomarkers"] }), _jsxs("h3", { className: "text-2xl md:text-3xl font-bold text-white mb-6", children: ["Distinguishing the ", _jsx("br", {}), _jsx("span", { className: "text-cyan-400", children: "Pathological Patterns" })] }), _jsxs("p", { className: "text-slate-300 leading-relaxed mb-6", children: ["Diabetic Retinopathy disrupts the retina's healthy vascular structure. The disease manifests as distinct visual anomalies\u2014", _jsx("strong", { children: "hard exudates" }), " (lipid deposits) and hemorrhages\u2014which create pixel-level contrasts against the retina."] }), _jsx("p", { className: "text-slate-400 text-sm mb-6 border-l-2 border-cyan-500 pl-4 italic", children: "\"It is this exact visual distinction that enables us to employ Computer Vision. By training KANTreeNet to recognize these specific textures and color gradients, we can automate screening with superhuman consistency.\"" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex gap-4 items-center", children: [_jsx("div", { className: "w-8 h-8 rounded bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400", children: _jsx("i", { className: "fas fa-check" }) }), _jsx("span", { className: "text-slate-300 text-sm", children: "Healthy: Clear vessels, uniform macula." })] }), _jsxs("div", { className: "flex gap-4 items-center", children: [_jsx("div", { className: "w-8 h-8 rounded bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400", children: _jsx("i", { className: "fas fa-times" }) }), _jsx("span", { className: "text-slate-300 text-sm", children: "Affected: Yellow lipid exudates, dark hemorrhages." })] })] })] }), _jsxs("div", { className: "relative h-full min-h-[400px] bg-slate-900 p-4 md:p-8 flex flex-col justify-center", children: [_jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { className: "relative aspect-square rounded-xl overflow-hidden group border border-emerald-500/30 shadow-lg shadow-emerald-900/20", children: [_jsx("img", { src: "https://upload.wikimedia.org/wikipedia/commons/7/75/Fundus_photograph_of_normal_right_eye.jpg", alt: "Normal Retina", className: "absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110", onError: (e) => {
                                                        // Fallback to abstract medical placeholder if blocked
                                                        e.currentTarget.src = "https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=600&auto=format&fit=crop";
                                                        e.currentTarget.style.filter = "hue-rotate(90deg) grayscale(50%)";
                                                    } }), _jsx("div", { className: "absolute bottom-0 w-full bg-slate-900/80 backdrop-blur-sm py-2 text-center", children: _jsx("span", { className: "text-xs font-bold text-emerald-400 uppercase tracking-widest", children: "Normal Retina" }) })] }), _jsxs("div", { className: "relative aspect-square rounded-xl overflow-hidden group border border-red-500/30 shadow-lg shadow-red-900/20", children: [_jsx("img", { src: "https://upload.wikimedia.org/wikipedia/commons/b/b1/Diabetic_retinopathy_-_fundus.jpg", alt: "Diabetic Retinopathy", className: "absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110", onError: (e) => {
                                                        // Fallback to abstract medical placeholder if blocked
                                                        e.currentTarget.src = "https://images.unsplash.com/photo-1579684385127-1ef15d508118?q=80&w=600&auto=format&fit=crop";
                                                    } }), _jsx("div", { className: "absolute top-[30%] right-[35%] w-12 h-12 rounded-full border-2 border-yellow-400/80 animate-[ping_2s_infinite]" }), _jsx("div", { className: "absolute bottom-[40%] left-[30%] w-8 h-8 rounded-full border-2 border-yellow-400/80 animate-[ping_2.5s_infinite]" }), _jsx("div", { className: "absolute bottom-0 w-full bg-slate-900/80 backdrop-blur-sm py-2 text-center", children: _jsx("span", { className: "text-xs font-bold text-red-400 uppercase tracking-widest", children: "Hard Exudates" }) })] })] }), _jsx("div", { className: "text-center mt-6", children: _jsxs("span", { className: "inline-block px-3 py-1 bg-slate-800 rounded-full text-xs text-slate-500 border border-slate-700", children: [_jsx("i", { className: "fas fa-robot mr-2 text-cyan-500" }), " AI Target Features"] }) })] })] }) }), _jsxs("div", { className: "text-center mb-10", children: [_jsx("h3", { className: "text-2xl font-bold text-white", children: "Why KANTreeNet?" }), _jsx("p", { className: "text-slate-400 text-sm mt-2", children: "Solving the limitations of traditional Deep Learning" })] }), _jsx("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-8", children: PROJECT_DETAILS.motivation.map((item, idx) => (_jsxs("div", { className: "p-8 rounded-2xl bg-slate-800 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-800/80 transition-all duration-300 group", children: [_jsx("div", { className: "w-14 h-14 rounded-xl bg-slate-700/50 flex items-center justify-center text-cyan-400 mb-6 group-hover:scale-110 group-hover:bg-cyan-500/10 transition-all", children: _jsx("i", { className: `fas ${item.icon} text-2xl` }) }), _jsx("h3", { className: "text-xl font-bold text-white mb-3", children: item.title }), _jsx("p", { className: "text-slate-400 leading-relaxed text-sm", children: item.content })] }, idx))) })] }) }));
const PIPELINE_STEPS = [
    {
        id: 1,
        title: "Original Fundus Image",
        subtitle: "Input Data",
        desc: "Original DR image read from disk.",
        details: "High-resolution retinal scan captured via fundus photography. Contains raw pixel data including noise and varying lighting conditions that must be processed.",
        type: "image",
        filter: "none"
    },
    {
        id: 2,
        title: "Resized to (448x448)",
        subtitle: "Preprocessing",
        desc: "Resized to model input size and normalized.",
        details: "Standardization of input dimensions ensures consistent feature extraction across different camera types. Color normalization reduces lighting variance.",
        type: "image",
        filter: "contrast(1.2) brightness(1.1)"
    },
    {
        id: 3,
        title: "Vessel Tree Branch",
        subtitle: "VesselTreeNet",
        desc: "Extract vessel-like structures from green channel, input to EnhancedVesselTreeNet.",
        details: "The green channel offers the best contrast for blood vessels. A specialized U-Net variant segments the vascular tree to identify structural anomalies.",
        type: "mask"
    },
    {
        id: 4,
        title: "Lesion Attention",
        subtitle: "Feature Extraction",
        desc: "ResNet high-level features + original image, automatically focus on suspected lesion regions.",
        details: "Class Activation Maps (CAM) highlight areas contributing most to the initial feature extraction, pinpointing potential microaneurysms and hemorrhages.",
        type: "heatmap-lesion"
    },
    {
        id: 5,
        title: "DAM Enhanced Local",
        subtitle: "Discriminative Attention",
        desc: "KANDAM module further enhances discriminative textures and local structures.",
        details: "Kolmogorov-Arnold Network Discriminative Attention Module (KANDAM) uses learnable activation functions to refine features and distinguish hard exudates from optic disc artifacts.",
        type: "heatmap-dam"
    },
    {
        id: 6,
        title: "ViT-S Global Context",
        subtitle: "Vision Transformer",
        desc: "Global attention heatmap based on ViT-S patch token norms.",
        details: "Vision Transformer (ViT-Small) processes the image as patches, capturing long-range dependencies and global retinal health context missed by local convolutions.",
        type: "heatmap-vit"
    },
    {
        id: 7,
        title: "Prediction: Severe",
        subtitle: "Differentiable Decision Tree",
        desc: "Final probability distribution across severity classes.",
        details: "The Differentiable Neural Decision Tree aggregates features from all branches to output a transparent, probabilistic diagnosis.",
        type: "chart"
    }
];
const Approach = () => {
    const [activeStep, setActiveStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    useEffect(() => {
        let interval;
        if (isPlaying) {
            interval = setInterval(() => {
                setActiveStep((prev) => (prev + 1) % PIPELINE_STEPS.length);
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);
    const step = PIPELINE_STEPS[activeStep];
    return (_jsx("section", { id: "approach", className: "py-24 bg-slate-950 relative overflow-hidden", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: [_jsxs("div", { className: "mb-12", children: [_jsx("h2", { className: "text-3xl md:text-4xl font-bold text-white mb-4", children: "How It Works: The Pipeline" }), _jsx("p", { className: "text-slate-400 text-lg max-w-2xl", children: "A step-by-step breakdown of how DR-KANTreeNet processes a retinal scan, from raw input to final diagnosis." })] }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12", children: [_jsx("div", { className: "lg:col-span-4 flex flex-col h-full", children: _jsxs("div", { className: "bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex-1 flex flex-col", children: [_jsxs("div", { className: "p-4 border-b border-slate-800 bg-slate-800/50 flex justify-between items-center", children: [_jsx("span", { className: "text-xs font-bold text-slate-400 uppercase tracking-widest", children: "Processing Steps" }), _jsxs("button", { onClick: () => setIsPlaying(!isPlaying), className: `text-xs px-3 py-1 rounded-full border transition-all ${isPlaying ? 'border-red-500 text-red-400 hover:bg-red-500/10' : 'border-emerald-500 text-emerald-400 hover:bg-emerald-500/10'}`, children: [_jsx("i", { className: `fas ${isPlaying ? 'fa-pause' : 'fa-play'} mr-1` }), " ", isPlaying ? 'Pause' : 'Auto Play'] })] }), _jsx("div", { className: "flex-1 overflow-y-auto", children: PIPELINE_STEPS.map((s, idx) => (_jsxs("button", { onClick: () => { setActiveStep(idx); setIsPlaying(false); }, className: `w-full text-left p-4 border-b border-slate-800/50 transition-all hover:bg-slate-800 ${activeStep === idx ? 'bg-slate-800/80 border-l-4 border-l-cyan-500' : 'border-l-4 border-l-transparent'}`, children: [_jsxs("div", { className: "flex justify-between items-start mb-1", children: [_jsxs("span", { className: `text-xs font-bold uppercase tracking-wider ${activeStep === idx ? 'text-cyan-400' : 'text-slate-500'}`, children: ["Step ", s.id] }), activeStep === idx && _jsx("i", { className: "fas fa-chevron-right text-cyan-500 text-xs animate-pulse" })] }), _jsx("h4", { className: `font-semibold ${activeStep === idx ? 'text-white' : 'text-slate-400'}`, children: s.title })] }, s.id))) })] }) }), _jsx("div", { className: "lg:col-span-8", children: _jsx("div", { className: "bg-black rounded-2xl border border-slate-700 shadow-2xl overflow-hidden relative aspect-[4/3] md:aspect-video flex flex-col", children: _jsxs("div", { className: "relative flex-1 bg-slate-900 flex items-center justify-center overflow-hidden", children: [step.type !== 'chart' && step.type !== 'mask' && (_jsx("img", { src: "https://upload.wikimedia.org/wikipedia/commons/b/b1/Diabetic_retinopathy_-_fundus.jpg", alt: "Retina Base", className: "absolute inset-0 w-full h-full object-cover transition-all duration-700", style: { filter: step.filter } })), step.type === 'mask' && (_jsx("div", { className: "absolute inset-0 bg-black", children: _jsx("img", { src: "https://upload.wikimedia.org/wikipedia/commons/b/b1/Diabetic_retinopathy_-_fundus.jpg", className: "w-full h-full object-cover grayscale contrast-[3.0] brightness-[1.5] invert mix-blend-screen opacity-90", style: { filter: 'contrast(10) grayscale(1)' } }) })), step.type.includes('heatmap') && (_jsx("div", { className: `absolute inset-0 opacity-60 mix-blend-overlay bg-gradient-to-tr transition-all duration-1000 
                    ${step.type === 'heatmap-lesion' ? 'from-transparent via-yellow-500 to-red-600' :
                                                step.type === 'heatmap-dam' ? 'from-blue-500 via-transparent to-emerald-500' :
                                                    'from-purple-600 via-red-400 to-transparent'}`, style: {
                                                backgroundSize: step.type === 'heatmap-vit' ? '20% 20%' : '100% 100%',
                                                backgroundImage: step.type === 'heatmap-vit' ? 'radial-gradient(circle, rgba(255,0,0,0.8) 0%, rgba(0,0,0,0) 70%)' : undefined
                                            } })), step.type === 'chart' && (_jsx("div", { className: "w-full h-full bg-slate-900 p-8 flex items-end justify-center gap-4 md:gap-8 pb-16", children: [
                                                { label: 'No DR', val: 18, h: '18%' },
                                                { label: 'Mild', val: 7, h: '7%' },
                                                { label: 'Moderate', val: 21, h: '21%' },
                                                { label: 'Severe', val: 36, h: '36%', active: true },
                                                { label: 'Proliferative', val: 16, h: '16%' }
                                            ].map((bar, i) => (_jsxs("div", { className: "flex flex-col items-center gap-2 group w-full max-w-[80px]", children: [_jsx("span", { className: `text-xs md:text-sm font-bold ${bar.active ? 'text-cyan-400' : 'text-slate-500'}`, children: bar.val / 100 }), _jsx("div", { className: "w-full bg-slate-800 rounded-t-lg relative overflow-hidden h-48 md:h-64 flex items-end", children: _jsx("div", { className: `w-full transition-all duration-1000 ${bar.active ? 'bg-cyan-500' : 'bg-slate-600'}`, style: { height: bar.h } }) }), _jsx("span", { className: "text-[10px] md:text-xs text-slate-400 text-center", children: bar.label })] }, i))) })), _jsx("div", { className: "absolute bottom-0 w-full bg-black/80 backdrop-blur-md p-6 border-t border-slate-700", children: _jsx("div", { className: "flex justify-between items-start", children: _jsxs("div", { children: [_jsxs("div", { className: "flex items-center gap-2 mb-1", children: [_jsx("span", { className: "text-cyan-400 text-xs font-bold uppercase", children: step.subtitle }), _jsx("span", { className: "text-slate-600 text-xs", children: "|" }), _jsxs("span", { className: "text-slate-400 text-xs", children: ["Frame 0", step.id] })] }), _jsx("h3", { className: "text-xl text-white font-bold mb-2", children: step.title }), _jsx("p", { className: "text-slate-300 text-sm max-w-3xl", children: step.desc }), _jsx("p", { className: "text-slate-500 text-xs mt-2 italic border-l-2 border-slate-700 pl-3", children: step.details })] }) }) }), _jsx("div", { className: "absolute top-0 left-0 w-full h-1 bg-slate-800", children: _jsx("div", { className: "h-full bg-cyan-500 transition-all duration-300", style: { width: `${((activeStep + 1) / PIPELINE_STEPS.length) * 100}%` } }) })] }) }) })] })] }) }));
};
const Implementation = () => (_jsx("section", { id: "implementation", className: "py-20 bg-slate-900", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: [_jsx("h2", { className: "text-3xl font-bold text-white mb-12 text-center", children: "Implementation Details" }), _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-8", children: [_jsxs("div", { className: "bg-[#1e1e1e] rounded-xl overflow-hidden shadow-2xl border border-slate-700 font-mono text-sm", children: [_jsxs("div", { className: "bg-[#252526] px-4 py-2 flex items-center gap-2 border-b border-black", children: [_jsx("div", { className: "w-3 h-3 rounded-full bg-red-500" }), _jsx("div", { className: "w-3 h-3 rounded-full bg-yellow-500" }), _jsx("div", { className: "w-3 h-3 rounded-full bg-green-500" }), _jsx("span", { className: "ml-2 text-slate-400 text-xs", children: "model.py" })] }), _jsx("div", { className: "p-4 text-slate-300 overflow-x-auto", children: _jsx("pre", { children: `class KANTreeNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # KAN Feature Extractor
        self.features = KANLayer(
            in_features=2048, 
            hidden=512
        )
        
        # Differentiable Tree Head
        self.tree = NeuralDecisionTree(
            depth=4, 
            classes=num_classes
        )

    def forward(self, x):
        features = self.features(x)
        return self.tree(features)` }) })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { className: "p-6 bg-slate-800 rounded-xl border border-slate-700", children: [_jsx("h3", { className: "text-white font-semibold mb-4", children: "Dataset" }), _jsxs("div", { className: "flex items-center gap-3 mb-2", children: [_jsx("i", { className: "fas fa-database text-cyan-400" }), _jsx("span", { className: "text-slate-300", children: "APTOS 2019" })] }), _jsx("p", { className: "text-sm text-slate-500", children: "3,662 retina images. Class imbalance handled via SMOTE." })] }), _jsxs("div", { className: "p-6 bg-slate-800 rounded-xl border border-slate-700", children: [_jsx("h3", { className: "text-white font-semibold mb-4", children: "Preprocessing" }), _jsxs("ul", { className: "text-sm text-slate-400 space-y-2", children: [_jsxs("li", { children: [_jsx("i", { className: "fas fa-angle-right text-emerald-400 mr-2" }), "Resize to 224x224"] }), _jsxs("li", { children: [_jsx("i", { className: "fas fa-angle-right text-emerald-400 mr-2" }), "CLAHE Enhancement"] }), _jsxs("li", { children: [_jsx("i", { className: "fas fa-angle-right text-emerald-400 mr-2" }), "Circle Crop"] })] })] }), _jsxs("div", { className: "col-span-2 p-6 bg-slate-800 rounded-xl border border-slate-700", children: [_jsx("h3", { className: "text-white font-semibold mb-4", children: "Stack" }), _jsx("div", { className: "flex gap-4 flex-wrap", children: PROJECT_DETAILS.stack.map(tech => (_jsx("span", { className: "px-3 py-1 bg-slate-700 rounded-full text-xs text-cyan-300 border border-slate-600", children: tech }, tech))) })] })] })] })] }) }));
const Results = () => (_jsx("section", { id: "results", className: "py-20 bg-slate-950", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: [_jsx("h2", { className: "text-3xl font-bold text-white mb-12 text-center", children: "Experimental Results" }), _jsx("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-12", children: [
                    { label: "Accuracy", value: PROJECT_DETAILS.results.accuracy, icon: "fa-bullseye", color: "text-cyan-400", border: "border-cyan-500/30" },
                    { label: "AUC Score", value: PROJECT_DETAILS.results.auc, icon: "fa-chart-area", color: "text-emerald-400", border: "border-emerald-500/30" },
                    { label: "F1 Score", value: PROJECT_DETAILS.results.f1, icon: "fa-balance-scale", color: "text-purple-400", border: "border-purple-500/30" },
                    { label: "Kappa", value: PROJECT_DETAILS.results.kappa, icon: "fa-check-double", color: "text-amber-400", border: "border-amber-500/30" },
                ].map((m, i) => (_jsxs("div", { className: `bg-slate-900 p-6 rounded-xl border ${m.border} text-center group hover:bg-slate-800 transition-colors`, children: [_jsx("div", { className: `text-3xl font-bold text-white mb-2 group-hover:scale-110 transition-transform inline-block`, children: m.value }), _jsxs("div", { className: `text-xs font-bold uppercase tracking-wider ${m.color}`, children: [_jsx("i", { className: `fas ${m.icon} mr-2` }), m.label] })] }, i))) }), _jsx("div", { className: "grid grid-cols-1 lg:grid-cols-3 gap-8", children: _jsxs("div", { className: "lg:col-span-3 bg-slate-900 rounded-xl border border-slate-800 p-6", children: [_jsxs("div", { className: "flex justify-between items-center mb-6", children: [_jsx("h3", { className: "text-xl font-semibold text-white", children: "Comparative Analysis on APTOS 2019" }), _jsx("span", { className: "text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded border border-slate-700", children: "Metrics in % (scaled)" })] }), _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "w-full text-left border-collapse", children: [_jsx("thead", { children: _jsxs("tr", { className: "border-b border-slate-800", children: [_jsx("th", { className: "py-4 px-4 text-slate-400 font-medium text-sm uppercase", children: "Method" }), _jsx("th", { className: "py-4 px-4 text-slate-400 font-medium text-sm uppercase text-right", children: "AUC" }), _jsx("th", { className: "py-4 px-4 text-slate-400 font-medium text-sm uppercase text-right", children: "ACC" }), _jsx("th", { className: "py-4 px-4 text-slate-400 font-medium text-sm uppercase text-right", children: "F1" }), _jsx("th", { className: "py-4 px-4 text-slate-400 font-medium text-sm uppercase text-right", children: "Kappa" })] }) }), _jsx("tbody", { children: PROJECT_DETAILS.results.comparison.map((row, i) => {
                                            const isBest = (metric) => row.bestMetrics && row.bestMetrics.includes(metric);
                                            return (_jsxs("tr", { className: `border-b border-slate-800 transition-colors hover:bg-slate-800/50 ${row.highlight ? 'bg-cyan-900/20 border-cyan-500/30' : ''}`, children: [_jsxs("td", { className: `py-4 px-4 ${row.highlight ? 'text-cyan-400 font-bold' : 'text-slate-300'}`, children: [row.model, row.highlight && _jsx("span", { className: "ml-2 text-[10px] bg-cyan-500 text-white px-1.5 py-0.5 rounded-full align-middle", children: "OURS" })] }), _jsx("td", { className: `py-4 px-4 text-right ${isBest('auc') ? 'text-cyan-400 font-bold text-lg' : (row.highlight ? 'text-slate-300' : 'text-slate-400')}`, children: row.auc }), _jsx("td", { className: `py-4 px-4 text-right ${isBest('acc') ? 'text-cyan-400 font-bold text-lg' : (row.highlight ? 'text-slate-300' : 'text-slate-400')}`, children: row.acc }), _jsx("td", { className: `py-4 px-4 text-right ${isBest('f1') ? 'text-cyan-400 font-bold text-lg' : (row.highlight ? 'text-slate-300' : 'text-slate-400')}`, children: row.f1 }), _jsx("td", { className: `py-4 px-4 text-right ${isBest('kappa') ? 'text-cyan-400 font-bold text-lg' : (row.highlight ? 'text-slate-300' : 'text-slate-400')}`, children: row.kappa })] }, i));
                                        }) })] }) }), _jsx("p", { className: "mt-4 text-xs text-slate-500 italic", children: "* Note: \"-\" indicates metric not reported in original paper. Comparisons sourced from recent literature." })] }) }), _jsxs("div", { className: "mt-16", children: [_jsx("h3", { className: "text-xl font-semibold text-white mb-6 text-center", children: "Explainability Visualization (Grad-CAM)" }), _jsx("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [1, 2, 3, 4].map((i) => (_jsxs("div", { className: "relative aspect-square rounded-lg overflow-hidden bg-black group", children: [_jsx("div", { className: "absolute inset-0 bg-slate-800 flex items-center justify-center", children: _jsx("i", { className: "fas fa-eye text-4xl text-slate-700" }) }), _jsx("div", { className: "absolute inset-0 bg-gradient-to-tr from-transparent via-red-500/30 to-yellow-500/40 opacity-75 mix-blend-overlay" }), _jsx("div", { className: "absolute bottom-0 w-full p-2 bg-black/60 backdrop-blur-sm text-center", children: _jsxs("span", { className: "text-xs text-white", children: ["Class ", i, ": Severity High"] }) })] }, i))) })] })] }) }));
const Footer = () => (_jsx("footer", { id: "discussion", className: "bg-slate-950 border-t border-slate-800 py-12", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 text-center", children: [_jsx("h2", { className: "text-2xl font-bold text-white mb-4", children: "Discussion & Future Work" }), _jsxs("p", { className: "max-w-3xl mx-auto text-slate-400 mb-8 leading-relaxed", children: ["The comparative analysis demonstrates that DR-KANTreeNet achieves robust performance, matching or exceeding strong baselines from recent years like TransMIL and MIL-ViT in terms of accuracy and F1 score. However, it still falls slightly short of the latest state-of-the-art models like MIL-ViTa in AUC and Kappa metrics.", _jsx("br", {}), _jsx("br", {}), "Crucially, while MIL-ViTa optimizes purely for predictive performance, DR-KANTreeNet prioritizes ", _jsx("strong", { children: "clinical interpretability" }), " via its differentiable decision tree head. Future work will aim to close this performance gap by incorporating more advanced attention mechanisms while retaining the transparency that clinicians require."] }), _jsxs("div", { className: "flex justify-center space-x-6", children: [_jsx("a", { href: "#", className: "text-slate-400 hover:text-white text-2xl", children: _jsx("i", { className: "fab fa-github" }) }), _jsx("a", { href: "#", className: "text-slate-400 hover:text-white text-2xl", children: _jsx("i", { className: "fas fa-file-pdf" }) }), _jsx("a", { href: "#", className: "text-slate-400 hover:text-white text-2xl", children: _jsx("i", { className: "fab fa-linkedin" }) })] }), _jsx("p", { className: "mt-8 text-slate-600 text-sm", children: "\u00A9 2024 DR-KANTreeNet Project. All rights reserved." })] }) }));
// --- AI Chat Widget ---
const ChatWidget = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'model', text: 'Hi! I am the AI assistant for DR-KANTreeNet. Ask me anything about the research!' }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);
    // Use a ref to keep the chat session instance across renders
    const chatRef = useRef(null);
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isOpen]);
    const handleSend = () => __awaiter(void 0, void 0, void 0, function* () {
        if (!input.trim() || loading)
            return;
        const userMsg = input;
        setInput("");
        setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
        setLoading(true);
        try {
            // Initialize the Chat session if it doesn't exist
            if (!chatRef.current) {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                chatRef.current = ai.chats.create({
                    model: "gemini-2.5-flash",
                    config: {
                        systemInstruction: SYSTEM_INSTRUCTION
                    }
                });
            }
            // Send the message using the correct SDK method
            const result = yield chatRef.current.sendMessage({ message: userMsg });
            // The response text is a property, not a method
            const response = result.text;
            setMessages(prev => [...prev, { role: 'model', text: response }]);
        }
        catch (error) {
            console.error("Chat Error:", error);
            setMessages(prev => [...prev, { role: 'model', text: "Sorry, I encountered an error connecting to the AI. Please check your network or API Key." }]);
        }
        finally {
            setLoading(false);
        }
    });
    return (_jsxs("div", { className: "fixed bottom-6 right-6 z-50 flex flex-col items-end", children: [isOpen && (_jsxs("div", { className: "mb-4 w-80 md:w-96 h-[500px] bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-10", children: [_jsxs("div", { className: "p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-cyan-400 animate-pulse" }), _jsx("span", { className: "font-semibold text-white", children: "Project Assistant" })] }), _jsx("button", { onClick: () => setIsOpen(false), className: "text-slate-400 hover:text-white", children: _jsx("i", { className: "fas fa-times" }) })] }), _jsxs("div", { className: "flex-1 overflow-y-auto p-4 space-y-4 bg-slate-900/50", ref: scrollRef, children: [messages.map((m, i) => (_jsx("div", { className: `flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`, children: _jsx("div", { className: `max-w-[85%] rounded-2xl px-4 py-2 text-sm ${m.role === 'user'
                                        ? 'bg-cyan-600 text-white rounded-br-none'
                                        : 'bg-slate-800 text-slate-200 rounded-bl-none'}`, children: m.text }) }, i))), loading && (_jsx("div", { className: "flex justify-start", children: _jsxs("div", { className: "bg-slate-800 rounded-2xl rounded-bl-none px-4 py-2 flex gap-1 items-center", children: [_jsx("div", { className: "w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" }), _jsx("div", { className: "w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce", style: { animationDelay: '0.1s' } }), _jsx("div", { className: "w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce", style: { animationDelay: '0.2s' } })] }) }))] }), _jsx("div", { className: "p-3 bg-slate-800 border-t border-slate-700", children: _jsxs("div", { className: "flex gap-2", children: [_jsx("input", { type: "text", value: input, onChange: (e) => setInput(e.target.value), onKeyDown: (e) => e.key === 'Enter' && handleSend(), placeholder: "Ask about accuracy, model...", className: "flex-1 bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500" }), _jsx("button", { onClick: handleSend, disabled: loading, className: "p-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-500 disabled:opacity-50", children: _jsx("i", { className: "fas fa-paper-plane" }) })] }) })] })), _jsx("button", { onClick: () => setIsOpen(!isOpen), className: "w-14 h-14 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/30 flex items-center justify-center text-white text-2xl hover:scale-105 transition-transform", children: _jsx("i", { className: `fas ${isOpen ? 'fa-times' : 'fa-robot'}` }) })] }));
};
const App = () => {
    return (_jsxs("div", { className: "min-h-screen", children: [_jsx(Navigation, {}), _jsx(Hero, {}), _jsx(Motivation, {}), _jsx(Approach, {}), _jsx(Implementation, {}), _jsx(Results, {}), _jsx(Footer, {}), _jsx(ChatWidget, {})] }));
};
const root = createRoot(document.getElementById("root"));
root.render(_jsx(App, {}));
