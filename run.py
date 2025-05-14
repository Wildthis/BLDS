from api.blds import app, blds

# 注册蓝图
# app.register_blueprint(blds, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 