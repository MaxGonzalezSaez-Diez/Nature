import os
import colorsys
import json
import numpy as np
import plotly.graph_objects as go
import panel as pn
from utils import utils as util
pn.extension()

class NatureVisualizer:
    """
    This class allows us to visualize a simulation using Plotly and Panel.
    It takes a JSON file with water, agent positions, and plant positions.
    """

    def __init__(self, json_file_path, full_checkpoint_dir, trace_length=2, max_iter=50):
        # with open(json_file_path, 'r') as f:
        #     self.data = json.load(f)

        self.data = []
        with open(json_file_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        
        max_iter_dict = self.data[-1][-1]['iter']
        self.max_iter = max_iter if max_iter < max_iter_dict else max_iter_dict

        self.all_species = list(set([i['species'] for i in self.data[1][0]['agents']]))
        
        self.trace_length = trace_length
        self.full_checkpoint_dir = full_checkpoint_dir
        
        self.all_markers = ['triangle-up', 'square', 'diamond', 'cross', 'pentagon']
        
        self.spe_to_markers = {'p': 'circle'}
        
        for single_sp in self.all_species:
            self.spe_to_markers[single_sp] = np.random.choice(self.all_markers)
        
        self.spe_to_color = self.assign_species_colors(self.all_species)
        
        self.process_data()

    def process_data(self):
        self.water_array = np.array(self.data[0])

        self.data_world = [j for i in self.data[1:] for j in i]
        
        self.agent_positions = {}
        self.plant_positions = {}
        for iter_data in self.data_world:
            iter_idx = int(iter_data['iter'])
            if iter_idx > self.max_iter:
                break

            self.agent_positions[iter_idx] = {}
            self.plant_positions[iter_idx] = {}
            for agent_data in iter_data['agents']:
                agent_id = agent_data['id']
                position = np.array(agent_data['position'])
                self.agent_positions[iter_idx][agent_id] = position
        
            for plant_data in iter_data['plants']:
                plant_id = plant_data['id']
                position = np.array(plant_data['position'])
                self.plant_positions[iter_idx][plant_id] = position

    def plot_single_rollout(self):
        print("Plotting single rollout")
        num_iters = self.max_iter
        
        iter_slider = pn.widgets.Player(
            name="Iteration", start=0, end=num_iters, value=0
        )
        iter_slider.interval = 200
        iter_slider.loop = True
        
        # Window size and position controls
        window_size = pn.widgets.IntSlider(name="Window Size", start=20, end=50, value=0, step=5)
        x_pos = pn.widgets.IntSlider(name="X Position", start=0, end=50, value=0)
        y_pos = pn.widgets.IntSlider(name="Y Position", start=0, end=50, value=0)
        
        @pn.depends(iter_slider.param.value, x_pos.param.value, y_pos.param.value, window_size.param.value)
        def create_plot(iteration, x_start, y_start, window_size):
            return self.get_single_frame(
                self.agent_positions, 
                self.plant_positions, 
                self.water_array, 
                iteration, 
                x_start, 
                y_start, 
                window_size
            )
        
        dashboard = pn.Column(
            pn.pane.Markdown("# Simulation Viewer", margin=(10, 10, 10, 10)),
            create_plot,
            pn.Row(
                iter_slider,
                pn.Column(
                    x_pos,
                    y_pos,
                    window_size
                )
            ),
            sizing_mode='stretch_width'
        )
        
        full_plot_path = f'{self.full_checkpoint_dir}/plot.html'
        os.makedirs(self.full_checkpoint_dir, exist_ok=True)
        dashboard.save(full_plot_path, embed=True)
        
        print(f"Visualization saved to {full_plot_path}")
        return dashboard

    def get_single_frame(self, agents_position, plants_position, water_array, iter_idx, x_start, y_start, window_size, tail_length=2):
        data = agents_position[iter_idx]
        data_plants = plants_position[iter_idx]
        
        x_end = x_start + window_size
        y_end = y_start + window_size
        limits = {'xs': x_start, 'xe': x_end, 'ys': y_start, 'ye': y_end}
        
        water = self.draw_water(water_array, limits)
        agents_plot = self.plot_agents(data, limits)
        tail_traces = self.plot_agent_tails(data, agents_position, tail_length, iter_idx, limits)
        plants_scatter = self.plot_plants(data_plants, limits)
        legend_items = self.plot_legend(data)
        grid_lines = self.plot_grid_lines(limits)
        
        fig = go.Figure(water + legend_items + plants_scatter + tail_traces + agents_plot + grid_lines)
        
        fig.update_layout(
            width=450,
            height=450,
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                range=[x_start - 0.5, x_end + 0.5],
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                range=[y_start - 0.5, y_end + 0.5],
                showticklabels=False,
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        fig.update_layout(modebar_remove=['pan', 'zoom'])
        
        return fig

    def in_limits(self, limits, x_val, y_val):
        return (limits['xs'] <= x_val < limits['xe']) and (limits['ys'] <= y_val < limits['ye'])

    def draw_water(self, water_array, limits):
        water_position = np.argwhere(water_array == 1)
        water_position = [pos for pos in water_position if self.in_limits(limits, pos[0], pos[1])]
        
        if not water_position:
            return []
        
        # Check which water cells are surrounded by other water cells
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        water_set = {tuple(pos) for pos in water_position}
        
        # S: "surrounded by water", N: "not surrounded by water"
        s_water = []
        n_water = []
        
        for pos in water_position:
            if all(((pos[0] + dx, pos[1] + dy) in water_set) for dx, dy in offsets):
                s_water.append(pos)
            else:
                n_water.append(pos)
        
        col_s = ['#004080' for _ in range(len(s_water))]  
        col_n = ['#7fbadc' for _ in range(len(n_water))] 
        
        colors = col_s + col_n
        water_p = s_water + n_water
        
        return [
            go.Scatter(
                x=[x - 0.5, x + 0.5, x + 0.5, x - 0.5, x - 0.5],
                y=[y - 0.5, y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=col,
                showlegend=False,
            )
            for (x, y), col in zip(water_p, colors)
        ]

    def plot_agents(self, data, limits):
        if not data:
            return []
        
        agents_scatter = []
        
        for agent_id, pos in data.items():
            species_id = util.strip_idNr(agent_id)
            
            if not self.in_limits(limits, pos[0], pos[1]):
                continue
            
            agents_scatter.append(
                go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=f'rgb{self.spe_to_color[species_id]}',
                        symbol=self.spe_to_markers[species_id],
                        line=dict(width=0)
                    ),
                    showlegend=False
                )
            )
        
        return agents_scatter

    def plot_agent_tails(self, data, agents_position, tail_length, iter_idx, limits):
        tail_traces = []
        
        for agent_id in data.keys():
            if iter_idx < 1:
                continue
            
            max_lookback = min(tail_length, iter_idx)
            
            pos = []
            for i in range(max_lookback, -1, -1):
                if iter_idx - i in agents_position and agent_id in agents_position[iter_idx - i]:
                    pos.append(agents_position[iter_idx - i][agent_id])
            
            if len(pos) <= 1:
                continue
            
            species_id = util.strip_idNr(agent_id)
            color = self.spe_to_color[species_id]
            
            valid_pos = []
            for i in range(len(pos)):
                if i == 0 or np.linalg.norm(pos[i] - pos[i-1]) <= 3*np.sqrt(2):
                    valid_pos.append(pos[i])
                else:
                    if valid_pos:
                        if any(self.in_limits(limits, p[0], p[1]) for p in valid_pos):
                            tail_traces.append(
                                go.Scatter(
                                    x=[p[0] for p in valid_pos],
                                    y=[p[1] for p in valid_pos],
                                    mode="lines",
                                    line=dict(color=f'rgb{color}', width=2, dash='dot'),
                                    showlegend=False
                                )
                            )
                    valid_pos = [pos[i]]
            
            if len(valid_pos) > 1 and any(self.in_limits(limits, p[0], p[1]) for p in valid_pos):
                tail_traces.append(
                    go.Scatter(
                        x=[p[0] for p in valid_pos],
                        y=[p[1] for p in valid_pos],
                        mode="lines",
                        line=dict(color=f'rgb{color}', width=2, dash='dot'),
                        showlegend=False
                    )
                )
        
        return tail_traces

    def plot_plants(self, data_plants, limits):
        if not data_plants:
            return []
        
        plant_positions = []
        
        for plant_id, pos in data_plants.items():
            if self.in_limits(limits, pos[0], pos[1]):
                plant_positions.append(pos)
        
        if not plant_positions:
            return []
        
        plant_color = 'rgb{}'.format(self.spe_to_color['p'])
        
        return [
            go.Scatter(
                x=[pos[0] for pos in plant_positions],
                y=[pos[1] for pos in plant_positions],
                mode="markers",
                marker=dict(
                    size=12,
                    color=plant_color,
                    symbol=self.spe_to_markers['p'],
                    line=dict(width=0)
                ),
                showlegend=False
            )
        ]

    def plot_legend(self, data):
        legend_items = []
        
        species_in_frame = set()
        for agent_id in data:
            species_id = util.strip_idNr(agent_id)
            species_in_frame.add(species_id)
        
        species_in_frame.add('p')
        
        for species_id in species_in_frame:
            symbol = self.spe_to_markers[species_id]
            color = 'rgb{}'.format(self.spe_to_color[species_id])
            
            if species_id == 'p':
                display_name = 'Plant'
            else:
                display_name = f'Species {species_id}'
            
            legend_items.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=color,
                        symbol=symbol,
                        line=dict(width=0)
                    ),
                    name=display_name
                )
            )
        
        return legend_items

    def plot_grid_lines(self, limits):
        x_start = limits['xs']
        x_end = limits['xe']
        y_start = limits['ys']
        y_end = limits['ye']
        
        grid_lines = []
        
        # Vertical grid lines
        for x_pos in range(x_start, x_end + 1):
            grid_lines.append(
                go.Scatter(
                    x=[x_pos - 0.5, x_pos - 0.5],
                    y=[y_start - 0.5, y_end + 0.5],
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                )
            )
        
        # Horizontal grid lines
        for y_pos in range(y_start, y_end + 1):
            grid_lines.append(
                go.Scatter(
                    x=[x_start - 0.5, x_end + 0.5],
                    y=[y_pos - 0.5, y_pos - 0.5],
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False,
                )
            )
        
        return grid_lines

    def assign_species_colors(self, species):
        # For each species let's just assign a random color to it. Higher up in the food chain should be more red though and lower more blue and the rest somewhat in the middle.
        levels = sorted([util.get_level_nr(s) for s in species])
        level_to_base_hue = {
            level: (0.17 - (i / max(1, len(levels) - 1)) * 0.17)
            for i, level in enumerate(levels)
        }

        # This just jitters the colors a little bit so that species on the same level are different but related. Doing it in hsv space cause it's easier to keep track of red, blue, etc.
        species_colors = {'p': (0, 0.6, 0)}
        for single_sp in species:
            level = util.get_level_nr(single_sp)
            sp = util.get_spe_nr(single_sp)

            base = level_to_base_hue[level]
            hue = max(0.0, min(0.17, base + sp * 0.01))

            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            species_colors[single_sp] = (r, g, b)

        return species_colors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=False)
    parser.add_argument('-j', '--json', type=str, required=False)
    parser.add_argument('-m', '--max_iter', type=int, required=False, default=10)
    parser.add_argument('--trace_length', type=int, default=2)
    
    args = parser.parse_args()

    if args.ckpt is None and args.json is None:
        raise ValueError
    
    json_path = args.json
    if args.json is not None:
        outputdir = os.path.dirname(os.path.abspath(json_path))
    else:
        outputdir = os.path.dirname(os.path.abspath(args.ckpt))
        json_path = [i for i in os.listdir(outputdir) if i.endswith('json')][0]
    
    visualizer = NatureVisualizer(
        json_file_path=json_path,
        full_checkpoint_dir=outputdir,
        trace_length=args.trace_length,
        max_iter=args.max_iter,
    )
    
    dashboard = visualizer.plot_single_rollout()