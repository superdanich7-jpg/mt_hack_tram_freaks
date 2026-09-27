FROM node:20-alpine AS build
WORKDIR /app

# Копируем манифесты и устанавливаем зависимости
COPY package*.json ./
RUN npm install

# Копируем исходный код и собираем продакшен-бандл
COPY . .
ENV VITE_API_BASE_URL=http://localhost:8000
RUN npm run build

# Раздаем статику через легковесный Nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Настройка SPA-роутинга для Nginx
RUN printf 'server {\n    listen 80;\n    location / {\n        root /usr/share/nginx/html;\n        index index.html;\n        try_files $uri $uri/ /index.html;\n    }\n}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]