import streamlit as st
import streamlit.components.v1 as components

def main():
    st.set_page_config(
            page_title="Inicio",
            page_icon='logo.png',
            layout="centered"
        )

    # Remove Streamlit's default padding and margin
    st.markdown(
        """
        <style>
            /* Remove top margin and padding of the main Streamlit section */
            .stMain {
                padding-top: 0rem !important;
                margin-top: 0rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Custom HTML with inline styling
    html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        /* Full-screen animated background */
        body {
            background-image: url('https://media4.giphy.com/media/xTiTnxpQ3ghPiB2Hp6/giphy.gif?cid=6c09b95219sdvq77x4l0mzc1omfnw4yceualg0y556obpxif&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g'); /* Replace with your animated image URL */
            background-size: cover;
            background-attachment: fixed;
            height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        /* Container for the title and divider */
        .container {
            position: relative;
            text-align: center;
            color: white;
            max-width: 1000px;
            width: 100%;
        }

        /* Horizontal divider line */
        .line {
            border-top: 2px solid white;
            width: 80%; /* Adjust width as needed */
            margin: 20px auto; /* Adds spacing above and below */
        }

        /* Big title text on top */
        .big-title {
            font-size: 50px;
            font-weight: bold;
            margin-bottom: 20px; /* Adds spacing below the title */
        }

        /* Small title text below */
        .small-title {
            font-size: 20px;
            margin-top: 20px; /* Adds spacing above the subtitle */
        }

    </style>
</head>
<body>
    <div class="container">
        <div class="big-title">EvoInvest</div>
        <div class="line"></div>
        <div class="small-title">Evoluciona tu portafolio con optimización financiera inteligente.</div>
    </div>
</body>
</html>

    """

    # Display the custom HTML in Streamlit
    components.html(html_code, height=350)

    cols = st.columns(3)
    with cols[0]:
        st.page_link("pages/popt.py", label="Genera Portafolios!", icon="📈")
    with cols[2]:
        st.page_link("pages/learn.py", label="Aprende Cómo Funciona", icon="🧠")


if __name__ == '__main__':
    main()