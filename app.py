# Copyright (c) 2026 Toby Medley. All rights reserved.
# Shared for portfolio/demonstration purposes only. Not licensed for reuse.

import streamlit as st

st.set_page_config(page_title="Markets & Risk Analytics Platform")

st.title("Markets & Risk Analytics Platform")

st.write("Use the sidebar to navigate between pages.")

with st.expander("ℹ️ About this project"):
    st.write(
        "Built by Toby Medley to demonstrate quantitative finance and Python skills "
        "for placement year applications. Shared publicly for portfolio purposes only, "
        "all rights reserved, not licensed for reuse or redistribution."
    )

st.divider()
st.caption("© 2026 Toby Medley. Shared for demonstration purposes only. All rights reserved.")