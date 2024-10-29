import vk_api
import networkx as nx
from pyvis.network import Network


def get_friends_list(vk, user_id):
    print(f'get friends list {user_id}')
    try:
        friends = vk.friends.get(user_id=user_id, fields='domain')
        lst = [friend_data['id'] for friend_data in friends["items"] if friend_data["first_name"] != "DELETED"]
        return lst
    except:
        print(f'User {user_id} was deleted or banned')
        return []


def create_graph(team_users_ids):
    with open("token.txt") as file:
        access_token = file.readline()
    session = vk_api.VkApi(token=access_token)
    vk = session.get_api()

    graph = {}

    for user_id in team_users_ids:
        if user_id in graph.keys():
            continue
        friends = get_friends_list(vk, user_id)
        graph[user_id] = friends
    
    friends_ids = []

    for cur_friends_ids in graph.values():
        friends_ids.extend(cur_friends_ids)
    friends_ids_set = set(friends_ids)

    for friend_id in friends_ids_set:
        friends = get_friends_list(vk, friend_id)
        graph[friend_id] = friends
    
    return graph


def visualize_graph(graph):
    net = Network()
    net.from_nx(graph)
    net.save_graph("graph.html")



if __name__ == '__main__':
    team_users_ids = [146450741, 150650454, 196656674, 211392424, 142332761]
    graph = create_graph(team_users_ids)
    with open("graph.txt", "w") as file:
        file.write(str(graph))
    
    with open("graph.txt", "r") as file:
        text = eval(file.read())
    
    graph = nx.from_dict_of_lists(text)
    print(graph)
    visualize_graph(graph)
