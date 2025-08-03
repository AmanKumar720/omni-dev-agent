from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
    
    print("\nOmni-Dev Agent session completed.")

if __name__ == "__main__":
    main()
