from sqlalchemy import create_engine
engine = create_engine('sqlite:///cookies.db')
engine1 = create_engine('sqlite:///:memory:')
engine2 = create_engine('sqlite://///home/cookiemonster/cookies.db')
engine3 = create_engine('sqlite:///c:\\Users\\cookiemonster\\cookies.db')

engine_mysql = create_engine('mysql+pymysql://cookiemonster:chocolatechip', '@mysql01.monster.internal/cookies', pool_recycle=3600)
