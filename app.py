import cohere
api_key="L41u8TnPpclKHjF0jJCxjD0SZ8O5yFQOaXoTibRL"
co = cohere.Client(api_key)
text ="""Tão natural quanto a luz do dia
Mas que preguiça boa, me deixa aqui à toa
Hoje ninguém vai estragar meu dia
Só vou gastar energia pra beijar sua boca

Fica comigo então, não me abandona, não
Alguém te perguntou como é que foi seu dia?
Uma palavra amiga, uma notícia boa
Isso faz falta no dia a dia
A gente nunca sabe quem são essas pessoas

Eu só queria te lembrar
Que aquele tempo eu não podia fazer mais por nós
Eu estava errado e você não tem que me perdoar
Mas também quero te mostrar
Que existe um lado bom nessa história
Tudo que ainda temos a compartilhar

E viver e cantar
Não importa qual seja o dia
Vamos viver, vadiar
O que importa é nossa alegria
Vamos viver e cantar
Não importa qual seja o dia
Vamos viver, vadiar
O que importa é nossa alegria

Tão natural quanto a luz do dia
Mas que preguiça boa, me deixa aqui à toa
Hoje ninguém vai estragar meu dia
Só vou gastar energia pra beijar sua boca

Eu só queria te lembrar
Que aquele tempo eu não podia fazer mais por nós
Eu estava errado e você não tem que me perdoar
Mas também quero te mostrar
Que existe um lado bom nessa história
Tudo que ainda temos a compartilhar

E viver e cantar
Não importa qual seja o dia
Vamos viver, vadiar
O que importa é nossa alegria
Vamos viver e cantar
Não importa qual seja o dia
Vamos viver, vadiar
O que importa é nossa alegria

Tão natural quanto a luz do dia"""
response = co.summarize(
    text=text,
    model='command',
    length='medium',
    extractiveness='medium'
)

summary = response.summary
print(summary)