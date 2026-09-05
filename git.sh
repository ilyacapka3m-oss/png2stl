
# один раз на компьютер: имя и e-mail (используйте свои)
git config user.name "ilyacapka3m"
git config user.email "ilyacapka3m@gmail.com"

git init
git add .
git commit -m "png2stl + workflow сборки exe"

# ВСТАВЬТЕ СЮДА URL из шага 2:
git remote add origin https://github.com/ilyacapka3m-oss/png2stl.git
git branch -M main
git push -u origin main
