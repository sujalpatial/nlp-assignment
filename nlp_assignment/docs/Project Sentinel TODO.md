# Project Sentinel TODO

## Backend Infrastructure
- [x] Implement PyTorch inference engine with OPT-1.3b model loading and forward hooks
- [x] Compute Mahalanobis distance using Ledoit-Wolf shrinkage covariance estimation
- [x] Implement Logit-Lens entropy tracking across all layers
- [x] Implement dynamic hallucination flagging (σ > 3.0 threshold)
- [x] Implement PCA-based steering vector for causal intervention
- [x] Build FastAPI backend with token generation endpoint
- [x] Implement WebSocket server for real-time activation streaming
- [x] Create Baseline Pass (unfiltered) and Intervention Pass (steered) logic
- [x] Implement semantic similarity scoring (e.g., cosine similarity between embeddings)

## Frontend Architecture
- [x] Set up elegant design system with Tailwind CSS 4 and color palette
- [x] Create main dashboard layout with header and content areas
- [x] Implement responsive grid for multi-pane layout
- [x] Set up WebSocket client for real-time data reception

## Visual Element A — The Pulse
- [x] Integrate Recharts for real-time line chart visualization
- [x] Create per-layer drift score visualization
- [x] Implement smooth animation and real-time data updates
- [x] Add layer labels and axis formatting
- [x] Style chart with refined colors and typography

## Visual Element B — Token Prediction Highlighting
- [x] Implement token rendering with color-coded status
- [x] Add Yellow highlighting for Warning tokens
- [x] Add Red highlighting for Predicted Hallucination tokens
- [x] Implement smooth transitions and visual feedback
- [x] Display "Potential Hallucination" label for flagged tokens

## Visual Element C — Before/After Dual-Pane
- [x] Create left pane: "Original Hallucinated Response" with drift heatmap
- [x] Create right pane: "Mechanistically Corrected Response"
- [x] Implement heatmap overlay using color gradients
- [x] Display semantic similarity score between responses
- [x] Add visual comparison indicators (e.g., diff highlights)

## Integration & Testing
- [x] Connect WebSocket client to backend streaming
- [ ] Test real-time data flow end-to-end
- [ ] Verify token flagging accuracy and timing
- [ ] Test causal intervention logic and steering effectiveness
- [ ] Validate semantic similarity calculations

## Polish & Refinement
- [x] Ensure typography is refined and consistent
- [x] Add micro-interactions (hover states, transitions, animations)
- [x] Refine color palette for visual elegance
- [x] Optimize spacing and layout for visual hierarchy
- [ ] Add loading states and error handling
- [ ] Ensure accessibility (WCAG compliance)
- [ ] Test cross-browser compatibility

## Documentation & Delivery
- [ ] Write API documentation for WebSocket protocol
- [ ] Document inference engine architecture
- [ ] Create user guide for dashboard features
- [ ] Prepare final checkpoint
