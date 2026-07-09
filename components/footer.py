"""
footer.py
=========

Professional Footer Component
for Smart City Traffic Forecasting Dashboard.

Author:
    Gaurav Kumar
"""

from datetime import datetime
import streamlit as st


# ==========================================================
# Footer
# ==========================================================

def render_footer():

    current_year = datetime.now().year

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 2])

    # ------------------------------------------------------
    # About
    # ------------------------------------------------------

    with col1:

        st.markdown("### 🚦 Smart City Traffic Forecasting")

        st.caption(
            """
            AI-powered traffic prediction system using
            Machine Learning.
            """
        )

    # ------------------------------------------------------
    # Technologies
    # ------------------------------------------------------

    with col2:

        st.markdown("### 🛠 Technologies")

        st.markdown("""
- Python
- Scikit-Learn
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
""")

    # ------------------------------------------------------
    # Author
    # ------------------------------------------------------

    with col3:

        st.markdown("### 👨‍💻 Developer")

        st.markdown("**Gaurav Kumar**")

        st.caption("AI / ML Engineer")

        st.markdown(
            "[📧 Email](mailto:gauravkumar902704@gmail.com)"
        )

        st.markdown(
            "[💻 GitHub](https://github.com/gauravkumar902704)"
        )

        st.markdown(
            "[🔗 LinkedIn](https://www.linkedin.com/in/gaurav-kumar-a756a1278?utm_source=share_via&utm_content=profile&utm_medium=member_android)"
        )

    st.markdown("---")

    st.markdown(
        f"""
<div class="footer">

Made with ❤️ using Streamlit & Scikit-Learn

<br>

© {current_year} Gaurav Kumar

</div>
""",
        unsafe_allow_html=True,
    )