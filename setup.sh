mkdir -p ~/.streamlit/
echo "\
[general]\n\email = luiz-dh@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableC0RS=false\n\
port = $P0RT\n\
" > ~/.streamlit/config.toml
