from finalytics import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(20), nullable=False)
    portfolios  = db.relationship('Portfolio', backref = 'creator', lazy = True)

    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable = False, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)

class Ticker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(6), nullable = False, unique=True)
    portfolio = db.Column(db.String(20), default= 'rosaster')
