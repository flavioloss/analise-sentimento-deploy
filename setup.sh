mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"flavioploss@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enablecors=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
