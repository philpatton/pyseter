import os

from PIL import Image
import pandas as pd

def launch_review(prediction_df, id_df, query_dir, reference_dir, top_k=5):
    
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the review app. Install it with: pip install gradio")
    

    def show_predictions(query_file, prediction_df, id_df, image_dir, reference_dir, top_k=5, max_imgs=10):
        """Return a list of (image, label) tuples for gr.Gallery."""
        preds = prediction_df[
            (prediction_df['image'] == query_file) &
            (prediction_df['predicted_id'] != 'new_individual')
        ].head(top_k)
        
        gallery_items = []
        
        # add query image
        query_path = os.path.join(image_dir, query_file)
        if os.path.exists(query_path):
            gallery_items.append((Image.open(query_path), f"QUERY: {query_file}"))
        
        # add reference images for each prediction
        for _, row in preds.iterrows():
            pid = row['predicted_id']
            score = row['score']
            ref_images = id_df[id_df['individual_id'] == pid]['image'].values[:max_imgs]
            for j, ref_file in enumerate(ref_images):
                ref_path = os.path.join(reference_dir, ref_file)
                if os.path.exists(ref_path):
                    gallery_items.append((Image.open(ref_path), f"{pid} | score: {score:.3f} | image {j+1}"))
        
        return gallery_items

    # --- Build the app ---
    query_files = prediction_df['image'].unique().tolist()

    def on_select(query_file):
        return show_predictions(query_file, prediction_df, id_df,
                                image_dir=query_dir,
                                reference_dir=reference_dir)
    with gr.Blocks() as demo:
        gr.Markdown("## AnyDorsal proposed IDs")
        
        index = gr.State(value=0)  # tracks current position
        confirmed = gr.State(value={})  # {query_file: predicted_id}

        label = gr.Markdown(f"**Image 1 / {len(query_files)}**")
        gallery = gr.Gallery(label="Proposed IDs", columns=2, height="auto")

        match_radio = gr.Radio(label="Select correct match", choices=[])
        
        with gr.Row():
            prev_btn = gr.Button("← Prev")
            next_btn = gr.Button("Next →")
            confirm_btn = gr.Button("✓ Confirm Match", variant="primary")
    
        with gr.Row():
            status = gr.Markdown("")
            download_btn = gr.Button("Download CSV")
            csv_file = gr.File(label="Confirmed matches")
        
        def navigate(idx, direction, confirmed):
            idx = idx + direction
            idx = max(0, min(idx, len(query_files) - 1))
            qf = query_files[idx]
            images = on_select(qf)
            
            # build radio choices from predictions
            preds = prediction_df[
                (prediction_df['image'] == qf) &
                (prediction_df['predicted_id'] != 'new_individual')
            ].head(5)
            choices = [f"{r['predicted_id']}" 
                    for _, r in preds.iterrows()]
            choices.append("new_individual")
            
            # check if already confirmed
            current = confirmed.get(qf, None)
            status_text = f"✓ Already confirmed: **{current}**" if current else ""
            
            return (idx, gr.Gallery(value=images, selected_index=0),
                    f"**{qf} — Image {idx+1} / {len(query_files)}**",
                    gr.Radio(choices=choices, value=None), status_text)
        def confirm(idx, selection, confirmed):
            if selection is None:
                return confirmed, "⚠️ No match selected"
            qf = query_files[idx]
            # extract just the id from the radio label
            pred_id = selection.split(" (score")[0]
            confirmed[qf] = pred_id
            return confirmed, f"✓ Confirmed: **{pred_id}** ({len(confirmed)} total)"

        def export_csv(confirmed):
            if not confirmed:
                return None
            df = pd.DataFrame(list(confirmed.items()), columns=['image', 'confirmed_id'])
            path = "confirmed_matches.csv"
            df.to_csv(path, index=False)
            return path

        prev_btn.click(fn=lambda idx, c: navigate(idx, -1, c), inputs=[index, confirmed],
                    outputs=[index, gallery, label, match_radio, status])
        next_btn.click(fn=lambda idx, c: navigate(idx, 1, c), inputs=[index, confirmed],
                    outputs=[index, gallery, label, match_radio, status])
        confirm_btn.click(fn=confirm, inputs=[index, match_radio, confirmed],
                        outputs=[confirmed, status])
        download_btn.click(fn=export_csv, inputs=confirmed, outputs=csv_file)

    demo.launch(inline=False)

def launch_quality_review(id_df, image_dir):
    """Launch app to select the best image per proposed_id / encounter."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required for the review app. Install it with: pip install gradio")

    grouped = id_df.groupby(['proposed_id', 'encounter'])
    
    # separate single-image batches (auto-confirm) from multi-image batches
    auto_confirmed = []
    review_batches = []
    
    for (pid, enc), group in grouped:
        images = group['image'].values.tolist()
        if len(images) == 1:
            auto_confirmed.append({'proposed_id': pid, 'encounter': enc, 'image': images[0]})
        else:
            review_batches.append({'proposed_id': pid, 'encounter': enc, 'images': images})
    
    with gr.Blocks() as demo:
        gr.Markdown("## Best Image Selector")
        gr.Markdown(f"**{len(auto_confirmed)}** single-image batches auto-confirmed. "
                    f"**{len(review_batches)}** batches to review.")
        
        index = gr.State(value=0)
        confirmed = gr.State(value={
            (d['proposed_id'], d['encounter']): d['image'] for d in auto_confirmed
        })
                
        label = gr.Markdown("")
        gallery = gr.Gallery(label="Images in this batch", columns=3, height="auto")
        
        with gr.Row():
            prev_btn = gr.Button("← Prev")
            next_btn = gr.Button("Next →")
        
        selected_radio = gr.Radio(label="Select best image", choices=[])
        confirm_btn = gr.Button("✓ Confirm Selection", variant="primary")
        status = gr.Markdown("")
        
        with gr.Row():
            download_btn = gr.Button("Download CSV")
            csv_file = gr.File(label="Selections")

        def navigate(idx, direction, confirmed):
            idx = idx + direction
            idx = max(0, min(idx, len(review_batches) - 1))
            batch = review_batches[idx]
            pid, enc, images = batch['proposed_id'], batch['encounter'], batch['images']
            
            gallery_items = []
            for img in images:
                img_path = os.path.join(image_dir, img)
                if os.path.exists(img_path):
                    gallery_items.append((Image.open(img_path), img))
            
            # check if already confirmed
            key = (pid, enc)
            current = confirmed.get(key, None)
            status_text = f"✓ Already selected: **{current}**" if current else ""
            
            return (idx,
                    gr.Gallery(value=gallery_items, selected_index=0),
                    f"**Batch {idx+1} / {len(review_batches)}** — {pid} / {enc} ({len(images)} images)",
                    gr.Radio(choices=images, value=current),
                    status_text)

        def confirm(idx, selection, confirmed):
            if selection is None:
                return confirmed, "⚠️ No image selected"
            batch = review_batches[idx]
            key = (batch['proposed_id'], batch['encounter'])
            confirmed[key] = selection
            total = len(confirmed)
            return confirmed, f"✓ Selected: **{selection}** ({total} / {len(review_batches) + len(auto_confirmed)} total)"

        def export_csv(confirmed):
            if not confirmed:
                return None
            rows = [{'proposed_id': k[0], 'encounter': k[1], 'image': v}
                    for k, v in confirmed.items()]
            df = pd.DataFrame(rows)
            path = "best_images.csv"
            df.to_csv(path, index=False)
            return path

        prev_btn.click(fn=lambda idx, c: navigate(idx, -1, c), inputs=[index, confirmed],
                       outputs=[index, gallery, label, selected_radio, status])
        next_btn.click(fn=lambda idx, c: navigate(idx, 1, c), inputs=[index, confirmed],
                       outputs=[index, gallery, label, selected_radio, status])
        confirm_btn.click(fn=confirm, inputs=[index, selected_radio, confirmed],
                          outputs=[confirmed, status])
        download_btn.click(fn=export_csv, inputs=confirmed, outputs=csv_file)
        
        # load first batch on startup
        demo.load(fn=lambda c: navigate(0, 0, c), inputs=confirmed,
                  outputs=[index, gallery, label, selected_radio, status])

    demo.launch(inline=False)