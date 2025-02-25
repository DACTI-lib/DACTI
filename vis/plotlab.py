import pyvista as pv
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
import glob
import colormaps
from PIL import Image
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

class plotlab:

    def __init__(
        self,
        model = "euler2d",
        dir_prj = os.getcwd(),
        dir_res = None,
        dir_out = None,
        case_name = None,
        c_map = "coolwarm",
        edge_color = "black",
        edge_width = 4,
        bound_width = 1.5,
        line_width = 1.2,
        font_size = 20,
        marker_size = 4,
        window_size = (4000, 4000)
    ):
        self.dir_prj        = dir_prj
        self.model          = model
        self.c_map          = c_map
        self.edge_color     = edge_color
        self.edge_width     = edge_width
        self.bound_width    = bound_width
        self.line_width     = line_width
        self.font_size      = font_size
        self.marker_size    = marker_size
        self.window_size    = window_size

        print("current project directory: ", dir_prj)

        # result folder
        if dir_res == None:
            self.dir_res = dir_prj + '/out/' + case_name + '/'
        else:
            self.dir_res = dir_res

        # output dir of figures/animation
        if dir_out == None:
            self.dir_out = dir_prj + '/vis/out/'
            if not os.path.exists(self.dir_out):
                os.makedirs(self.dir_out)
        else:
            self.dir_out = dir_out

        # zoomed in view
        self.zoomedin_view = False

        #####
        # General plot settings
        #####
        plt.rcParams['text.usetex'] = True
        plt.rcParams["font.family"] = "Times New Roman"
        pv.global_theme.nan_color = 'white'

    def to_label_name(self, name):
        if self.model == "euler2d":
            if name == "p_0": return r"$\rho~$[kg/m$^3$]"
            if name == "p_1": return r"$u~$[m/s]"
            if name == "p_2": return r"$v~$[m/s]"
            if name == "p_3": return r"$p~$[bar]"
        
        if self.model == "multi1d":
            if name == "p_0": return r"$\rho~$[$\times 10^3$ kg/m$^3$]"
            if name == "p_1": return r"$\rho_2(1-\alpha)$"
            if name == "p_2": return r"$u~$[m/s]"
            if name == "p_3": return r"$p~$[MPa]"
            if name == "p_4": return r"$\alpha$"

        if self.model == "multi2d":
            if name == "p_0": return r"$\rho_1\alpha$"
            if name == "p_1": return r"$\rho_2(1-\alpha)$"
            if name == "p_2": return r"$u~$[m/s]"
            if name == "p_3": return r"$v~$[m/s]"
            if name == "p_4": return r"$p~$[Pa]"
            if name == "p_5": return r"$\alpha$"
        
        if self.model == "multi3d":
            if name == "p_0": return r"$\rho_1\alpha$"
            if name == "p_1": return r"$\rho_2(1-\alpha)$"
            if name == "p_2": return r"$u~$[m/s]"
            if name == "p_3": return r"$v~$[m/s]"
            if name == "p_4": return r"$w~$[m/s]"
            if name == "p_5": return r"$p~$[Pa]"
            if name == "p_6": return r"$\alpha$"

        if name == "acti_level": return "Time level $L$"
        if name == "Ma": return "$Ma$"
        if name == "vorticity": return "$\omega_z$"
        if name == "qcriterion": return "$Q$"
        if name == "cfl": return "$\Delta t_I / \Delta t_I^{\max}$"
        if name == "T": return "$T$"
        if name == "ke": return "$ke$"

        return name
    
    def add_zoomedin_view(self, zoomedin_box, position, size, indication_lines=False):
        self.zoomedin_view = True
        self.zoomedin_box = zoomedin_box
        self.zoomedin_pos = position
        self.zoomedin_size = size
        self.zoomedin_indication_lines = indication_lines

        self.zoomedin_bound = [
            self.zoomedin_box[0, 0], self.zoomedin_box[1, 0],
            self.zoomedin_box[0, 1], self.zoomedin_box[1, 1],
            self.zoomedin_box[0, 2], self.zoomedin_box[1, 2]]

    def plot1d(
        self, 
        file, 
        case_name,
        scalar_name, 
        plotref=False, 
        plotinit=False,
        sideplot=False,
        file_ref=None,
        img_top=False,
        show_legend=False
    ):
        mesh = pv.read(file)
        mesh = mesh.point_data_to_cell_data()

        x = np.array(mesh.cell_centers().points[:, 0])

        def compute_T(mesh):
            gamma1, gamma2 = 6.12, 1.4
            P1, P2, cp1, cp2 = 343000000.0, 0.0, 4182, 1005
            cv1, cv2 = cp1 / gamma1, cp2 / gamma2

            alpha = np.array(mesh.cell_data["p_4"])
            p = np.array(mesh.cell_data["p_3"])
            r1 = np.array(mesh.cell_data["p_0"])
            r2 = np.array(mesh.cell_data["p_1"])
            r = r1 + r2

            gamma = 1 + 1 / (alpha/(gamma1-1) + (1-alpha)/(gamma2-1))
            P0 = (alpha*gamma1*P1/(gamma1-1) + (1-alpha)*gamma2*P2/(gamma2-1)) / (1 + alpha/(gamma1-1) + (1-alpha)/(gamma2-1))
            T = (p/(gamma-1) + alpha*P1/(gamma1-1) + (1-alpha)*P2/(gamma2-1)) / (r1*cv1 + r2*cv2)
            # T = (p + P0) / (gamma - 1) / (alpha*cv1 + (1-alpha)*cv2) / r
            # T = (p / (gamma - 1) + alpha*P1/(gamma1-1) + (1-alpha)*P2/(gamma2-1)) / (r1*cv1 + r2*cv2)

            return T
               
        if (scalar_name == "T"):
            scalar = compute_T(mesh)
        elif (scalar_name == "p_0"):
            scalar = np.array(mesh.cell_data[scalar_name]) / 1000
        elif (scalar_name == "p_3"):
            scalar = np.array(mesh.cell_data[scalar_name]) / 1e6
        else:
            scalar = np.array(mesh.cell_data[scalar_name])
        
        alpha = np.array(mesh.cell_data["p_4"])
        x_1 = x[alpha > 0.99]
        x_2 = x[alpha < 0.01]
        print(min(x_2))
        print(max(x_1))
        print(min(x_2) - max(x_1))
        
        fig, ax = None, None
        if img_top:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)
            ax_top = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
            self.format_ax(ax_top)
            self.format_ax(ax)
            ax.plot(x, scalar, 'o', markersize=self.marker_size, markerfacecolor='none', color='black', label="numerical")
            ax.set_xlabel(r'$x~$[m]', fontsize=self.font_size)
            ax.set_ylabel(self.to_label_name(scalar_name), fontsize=self.font_size)

            ax_top.plot(x, alpha, 'o', markersize=self.marker_size, markerfacecolor='none', color='black', label="numerical")
            ax_top.set_xticklabels([])
            ax_top.set_ylabel(self.to_label_name("p_4"), fontsize=self.font_size)
        else:
            fig, ax = plt.subplots()
            self.format_ax(ax)
            ax.plot(x, scalar, 'o', markersize=self.marker_size, markerfacecolor='none', color='black', label="numerical")
            ax.set_xlabel(r'$x~$[m]', fontsize=self.font_size)
            ax.set_ylabel(self.to_label_name(scalar_name), fontsize=self.font_size)

        if plotref:
            data = np.loadtxt(self.dir_prj + '/vis/' + file_ref)
            x_ref = np.array(data[0, :]) - 1.0

            if scalar_name == "p_0": ax.plot(x_ref, data[1, :]/1000, 'k-', linewidth=self.line_width, label="analytical")
            if scalar_name == "p_2": ax.plot(x_ref, data[2, :], 'k-', linewidth=self.line_width, label="analytical")
            if scalar_name == "p_3": ax.plot(x_ref, data[3, :]/1e6, 'k-', linewidth=self.line_width, label="analytical")
            if scalar_name == "p_4": ax.plot(x_ref, data[4, :], 'k-', linewidth=self.line_width, label="analytical")
            if scalar_name == "T": ax.plot(x_ref, data[5, :], 'k-', linewidth=self.line_width, label="analytical")
            if scalar_name == "acti_level": 
                ax2 = ax.twinx()
                ax2.plot(x, alpha, 'o', markersize=self.marker_size)

            if img_top:
                ax_top.plot(x_ref, data[4, :], 'k-', linewidth=self.line_width, label="analytical")

        if plotinit:
            mesh = pv.read(self.dir_res + case_name + '_0000.vtu')
            mesh = mesh.point_data_to_cell_data()

            x = np.array(mesh.cell_centers().points[:, 0])
            if (scalar_name == "T"):
                scalar = compute_T(mesh)
            elif (scalar_name == "p_0"):
                scalar = np.array(mesh.cell_data[scalar_name]) / 1000
            elif (scalar_name == "p_3"):
                scalar = np.array(mesh.cell_data[scalar_name]) / 1e6
            else:
                scalar = np.array(mesh.cell_data[scalar_name])
                
            alpha  = np.array(mesh.cell_data["p_4"])
            indices = np.argsort(x)
            x      = x[indices]
            scalar = scalar[indices]
            alpha  = alpha[indices]

            ax.plot(x, scalar, 'k--', linewidth=self.line_width, label="initial")
            if img_top:
                ax_top.plot(x, alpha, 'k--', linewidth=self.line_width, label="initial")
        
        if show_legend:
            ax.legend(loc='best', frameon=False, fontsize=self.font_size+2)

        fig_name = self.dir_out + f"{case_name}_{scalar_name}.pdf"
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
        plt.show()

    def plotObservables(
        self, 
        case_name, 
        scalar_name, 
        showfig=True,
        savefig=True,
        animation=False,
        startframe=0
    ):
        data = self.load_observables(case_name)

        time = data["time"]
        scalar = data[scalar_name]

        time = time.to_numpy()
        scalar = scalar.to_numpy()
        time   =   time[startframe:]
        scalar = scalar[startframe:]

        fft_result = np.fft.fft(scalar - np.mean(scalar))
        n = len(scalar)
        freq = np.fft.fftfreq(n, d=time[1]-time[0])
        freq = freq[:n // 2]
        magn = np.abs(fft_result[:n // 2])

        if not animation:
            # plot the original signal and fft results
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            self.format_ax(ax[0])
            self.format_ax(ax[1])
            ax[0].plot(time, scalar, 'k-', linewidth=self.line_width)
            ax[0].set_xlabel(r"$t$", size=self.font_size)
            ax[0].set_ylabel(self.to_label_name(scalar_name), size=self.font_size)
            ax[1].semilogy(freq, magn, 'k-', linewidth=self.line_width)
            freq_main = freq[np.argmax(magn)]
            ax[1].vlines(freq_main, np.min(magn), np.max(magn), 'r', '--', linewidth=1.2)
            ax[1].text(freq_main, np.min(magn), f"{freq_main} Hz", fontsize=self.font_size, color='r')

            fig_name = self.dir_out + f"{case_name}_{scalar_name}.pdf"

            fig.tight_layout()
            if showfig: plt.show()
            if savefig: 
                fig.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
                plt.close()
                print(f'Result figure(s) saved to {fig_name}')
        else:
            i = 0
            for t in time:
                fig, ax = plt.subplots()
                self.format_ax(ax)
                ax.plot(t, scalar[i], 'ro', markersize=self.marker_size)
                ax.plot(time, scalar, 'k-', linewidth=self.line_width)
                ax.set_xlabel(r"$t$", size=self.font_size)
                ax.set_ylabel(r"$ke$", size=self.font_size)

                i += 1
                frame = str(i).zfill(4)
                fig_name = self.dir_out + f"{case_name}_{scalar_name}_{frame}.png"

                fig.tight_layout()
                fig.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
                plt.close()
                print(f'Result figure(s) saved to {fig_name}')

        return fig_name

    def plot2d(
        self,
        file, 
        scalar_name=None, 
        plottype=None,
        v_min=None, 
        v_max=None, 
        showgrid=False,
        showfig=True, 
        savefig=False,
        hide_axes=False,
        hide_cbar=False,
        hide_spines=False,
        int_cbar=False,
        log_scale=False,
        cbar_orientation="vertical",
        zoomedin_box=np.array([]),
        savetype='pdf',
        figsize=(6, 6)
    ):  
        img_top = None
        if plottype == "half_schlieren":
            img_top, ex = self.pvplot2d(file, None, "schlieren", v_min, v_max, False, log_scale, zoomedin_box)
        elif plottype == "half_grid":
            img_top, ex = self.pvplot2d(file, None, None, v_min, v_max, True, log_scale, zoomedin_box)

        img, extent = self.pvplot2d(file, scalar_name, plottype, v_min, v_max, showgrid, log_scale, zoomedin_box)
        
        img_zoomedin = None
        if self.zoomedin_view:
            img_zoomedin, ex = self.pvplot2d(file, scalar_name, plottype, v_min, v_max, showgrid, log_scale, self.zoomedin_box)
        
        # wrap with matplotlib
        fig_name = self.matplotlib_wrapper(
            file, 
            img, img_top,
            extent, 
            img_zoomedin=img_zoomedin, 
            savefig=savefig, 
            showfig=showfig, 
            scalar_name=scalar_name, 
            vmin=self.v_min, 
            vmax=self.v_max,
            hide_axes=hide_axes,
            hide_cbar=hide_cbar,
            hide_spines=hide_spines,
            int_cbar=int_cbar,
            log_scale=log_scale,
            cbar_orientation=cbar_orientation,
            savetype=savetype,
            figsize=figsize
        )
        if savefig: print(f'Result figure(s) saved to {fig_name}')

    def pvplot2d(
        self, 
        file, 
        scalar_name, 
        plottype,
        v_min, 
        v_max, 
        showgrid, 
        log_scale,
        zoomedin_box
    ):
        mesh = pv.read(file)

        u = np.array(mesh.point_data["p_1"])
        v = np.array(mesh.point_data["p_2"])
        mesh["velocity"] = np.column_stack((u, v, np.zeros_like(u)))

        if plottype == "schlieren" :
            mesh = mesh.compute_derivative(scalars="p_0")
            drho = np.array(mesh["gradient"])
            drho_norm = np.abs(np.linalg.norm(drho, axis=1))

            k = 50
            norm_max = np.max(drho_norm)
            mesh["schlieren"] = np.exp(-k * drho_norm / norm_max)
            mesh = mesh.point_data_to_cell_data()

        if scalar_name != None:
            if scalar_name == "Ma":
                mesh = mesh.point_data_to_cell_data()
                rho = np.array(mesh.cell_data["p_0"])
                u = np.array(mesh.cell_data["p_1"])
                v = np.array(mesh.cell_data["p_2"])
                p = np.array(mesh.cell_data["p_3"])
                a = np.sqrt(1.4 * p / rho)
                mach = np.sqrt(u**2 + v**2) / a
                mesh[scalar_name] = mach
                if v_min==None : v_min = np.min(mach)
                if v_max==None : v_max = np.max(mach)
            elif scalar_name == "vorticity" or scalar_name == "qcriterion":
                mesh = mesh.compute_derivative(scalars='velocity', vorticity=True, qcriterion=True)
                mesh["vorticity"] = mesh["vorticity"][:,2]
                if v_min==None : v_min = np.min(np.array(mesh[scalar_name]))
                if v_max==None : v_max = np.max(np.array(mesh[scalar_name]))
            else:
                mesh = mesh.point_data_to_cell_data()
                scalar_field = np.array(mesh.cell_data[scalar_name])
                if v_min==None : v_min = np.min(scalar_field)
                if v_max==None : v_max = np.max(scalar_field)

                if scalar_name == "cfl": 
                    mesh[scalar_name] = scalar_field / v_max
                    v_max = 1.0
            
            self.v_min  = v_min
            self.v_max  = v_max

            if log_scale:
                mesh[scalar_name] = np.log10(mesh[scalar_name])
                v_min, v_max = np.log10(v_min), np.log10(v_max)

        # zoomed in view
        if zoomedin_box.size != 0:
            if scalar_name == None:
                v_lower = zoomedin_box[0, :]
                v_upper = zoomedin_box[1, :]
            else:
                v_lower = self.__find_closest_vertex__(mesh, zoomedin_box[0, :])
                v_upper = self.__find_closest_vertex__(mesh, zoomedin_box[1, :])

            clip_bound = [v_lower[0], v_upper[0], v_lower[1], v_upper[1], v_lower[2], v_upper[2]]
            mesh = mesh.clip_box(bounds=clip_bound, invert=False)

        plotter = pv.Plotter(lighting="three lights", off_screen=True, window_size=self.window_size)
        
        # scalar plot
        if plottype == "schlieren":
            plotter.add_mesh(
                mesh, 
                scalars="schlieren", 
                cmap="gray", 
                show_scalar_bar=False)
        else:
            if scalar_name != None:
                plotter.add_mesh(
                    mesh,
                    scalars=scalar_name,
                    clim=(v_min, v_max),
                    cmap=self.c_map,
                    show_scalar_bar=False)
        
        # grid plot
        if showgrid:
            edges = self.__filter_edge__(mesh)
            plotter.add_mesh(edges, color=self.edge_color, line_width=self.edge_width)

        self.__adjust_camera__(plotter, '2d')

        # save screenshot
        img = plotter.screenshot(None, transparent_background=False, return_img=True)
        pv.close_all()

        return img, mesh.bounds

    def pvplot3d(
        self, 
        file, 
        scalar_name=None,
        v_min=None,
        v_max=None,
        plottype=None,
        showfig=False,
        showgrid=False,
        savefig=False,
        outline=False,
        savetype="png",
        clipbox=np.array([])
    ):
        mesh = pv.read(file)

        if scalar_name != None:
            if v_min==None or v_max==None:
                mesh_ = mesh.point_data_to_cell_data()
                scalar_field = np.array(mesh_.cell_data[scalar_name])
                if v_min==None : v_min = np.min(scalar_field)
                if v_max==None : v_max = np.max(scalar_field)

        if clipbox.size != 0:
            v_lower = clipbox[0, :]
            v_upper = clipbox[1, :]

            clip_bound = [v_lower[0], v_upper[0], v_lower[1], v_upper[1], v_lower[2], v_upper[2]]
            mesh = mesh.clip_box(bounds=clip_bound, invert=True)

        plotter = pv.Plotter(lighting="three lights", off_screen=not showfig, window_size=(3840, 2160))

        if scalar_name!=None:
            # contour plot
            if plottype == "contour":
                contour = mesh.contour(isosurfaces=[1.0], scalars=scalar_name)
                plotter.add_mesh(
                    contour, 
                    opacity=0.5, 
                    cmap=self.c_map,
                    show_scalar_bar=False
                )
                plotter.camera.position = (-1.0, -4.0, 0.0)
                plotter.camera.focal_point = (-1.0, 0, 0)
            # normal plot
            else:
                plotter.add_mesh(
                    mesh, 
                    scalars=scalar_name, 
                    clim=(v_min, v_max),
                    cmap=self.c_map,
                    show_scalar_bar=False
                )
                plotter.camera.position = (0.0, -2.8, 2.0)
                plotter.camera.focal_point = (0.0, -0.2, 0)
                plotter.camera.zoom(0.65)
                light = pv.Light(position=plotter.camera.position)
                light.intensity = 0.2
                plotter.add_light(light)

                if outline:
                    plotter.add_mesh(mesh.outline(), color="black", line_width=8)

        if showgrid or scalar_name==None:
            edges = self.__filter_edge__(mesh)
            plotter.add_mesh(edges, color=self.edge_color, line_width=self.edge_width)

        plotter.show()

        case_name = self.__file_to_case_name__(file)
        frame = self.__file_to_frame__(file)
        if scalar_name == None:
            img_name = self.dir_out + f"{case_name}_grid_{frame}.{savetype}"
        else:
            img_name = self.dir_out + f"{case_name}_{scalar_name}_{frame}.{savetype}"
        img = plotter.screenshot(img_name, transparent_background=True, return_img=True)
        pv.close_all()

        # if scalar_name!=None:
        #     fig, ax = plt.subplots()
        #     im = ax.imshow(img, cmap=self.c_map, extent=mesh.bounds, vmin=v_min, vmax=v_max, interpolation='none')

        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     for spine in ax.spines.values():
        #         spine.set_visible(False)

        #     div = make_axes_locatable(ax)
        #     cax = div.append_axes("bottom", size=0.15, pad=0.1)
        #     cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.5)
        #     cbar.set_label(self.to_label_name(scalar_name), size=self.font_size)
        #     cbar.ax.tick_params(direction='in')

        #     if savefig: 
        #         fig.savefig(img_name, dpi=300, bbox_inches="tight", transparent=True)
        #         plt.close()

        return img_name

    def animation(
        self,
        case_name, 
        scalar_name,
        plotdim,
        framerate=15,
        plottype=None, 
        v_min=None, 
        v_max=None, 
        showgrid=False, 
        keepfigs=False,
        hideaxes=False,
        hidecbar=False,
        create_videos=True,
        skip_plotting=False,
        zoomedin_box=np.array([])
    ):
        if not skip_plotting:
            # find all case results
            all_case_files = glob.glob(self.dir_res + f'{case_name}*.vtu')

            frames = np.array([self.__file_to_frame__(file) for file in all_case_files])
            frames_num = np.array(frames).astype(int)
            sorted_idx = np.argsort(frames_num)

            # generate figures
            figures = []
            for frame in frames[sorted_idx]:
                file = self.find_file(case_name, frame)

                if plotdim=='2d':
                    figures.append(
                        self.plot2d(
                            file, 
                            scalar_name,
                            plottype, 
                            v_min, v_max, 
                            showgrid, 
                            showfig=False,
                            savefig=True, 
                            hide_axes=hideaxes,
                            hide_cbar=hidecbar,
                            zoomedin_box=zoomedin_box,
                            savetype='png'
                        )
                    ) 
                elif plotdim=='3d':
                    figures.append(
                        self.pvplot3d(file, scalar_name, v_max=v_max, v_min=v_min, plottype=plottype)
                    )


        # create video
        if create_videos:
            prefix = self.dir_out + case_name 
            if scalar_name != None: prefix = prefix + '_' + scalar_name 
            video_name = prefix + '.mp4'
            os.system(
                " ".join(
                    [
                        "ffmpeg -y -loglevel warning",
                        f"-framerate {framerate}",
                        f"-i {prefix}_%04d.png",
                        "-pix_fmt yuv420p",
                        # "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2'",
                        "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2'",
                        "-r 60",
                        f"{video_name}",
                    ]
                )
            )

            print(f'video saved to {video_name}') 

        if not keepfigs:
            for fig in glob.glob(os.path.join(self.dir_out, '*.png')): 
                os.remove(fig) 
        
    def histogram(
        self,
        file,
        case_name,
        bins=10,
        showfig=True,
        savefig=True
    ):
        mesh = pv.read(file)
        mesh = mesh.point_data_to_cell_data()
        levels = np.array(mesh.cell_data['acti_level'])

        vmin, vmax = np.min(levels), np.max(levels)
        bin_centers = np.arange(vmin, vmax+1)
        bin_width = bin_centers[1] - bin_centers[0]
        bin_edges = np.append(bin_centers - bin_width / 2, bin_centers[-1] + bin_width / 2)

        cmap = plt.cm.get_cmap(self.c_map, len(bin_centers))
        norm = mpl.colors.BoundaryNorm(bin_centers, cmap.N)
        colors = [cmap(norm(i)) for i in bin_centers]

        fig, ax = plt.subplots()
        self.format_ax(ax)
        counts, binedges, patches = ax.hist(levels, bins=bin_edges, edgecolor="black", density=True)
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        ax.set_xlabel(self.to_label_name('acti_level'), size=self.font_size)
        ax.set_ylabel(r"$\frac{N_L}{N_{total}}$", size=self.font_size)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_centers, fontsize=self.font_size)
        formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x)}')
        ax.xaxis.set_major_formatter(formatter)

        fig_name = self.dir_out + f"{case_name}_level_histogram.pdf"
        fig.tight_layout()
        if showfig: plt.show()
        if savefig: 
            fig.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
            plt.close()
            print(f'Result figure(s) saved to {fig_name}')

        return fig

    def matplotlib_wrapper(
        self,
        file, 
        img, img_top,
        extent,
        img_zoomedin=None,
        savefig=True, 
        showfig=True,
        scalar_name=None,
        vmin=None, 
        vmax=None,
        hide_axes=False,
        hide_cbar=False,
        hide_spines=False,
        int_cbar=False,
        log_scale=False,
        cbar_orientation="vertical",
        savetype='pdf',
        figsize=(6, 6)
    ):
        case_name = self.__file_to_case_name__(file)
        frame = self.__file_to_frame__(file)
        fig_name = None
        
        fig, ax = None, None
        if img_top is not None:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.103], hspace=0)
            ax_top = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
            self.format_ax(ax_top, hide_spines=hide_spines)
            self.format_ax(ax, hide_spines=hide_spines)
            ax_top.imshow(img_top, extent=extent, interpolation='none')
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.set_ylim([0.0, extent[3]])
            ax.set_ylim([extent[2], 0.0])

        else:
            fig, ax = plt.subplots(figsize=figsize)
            self.format_ax(ax, hide_spines=hide_spines)

        if scalar_name == None:
            ax.imshow(img, extent=extent, interpolation='none')
            fig_name = self.dir_out + case_name + '_' + frame + '.' + savetype

        else:
            im = None
            color_levels = []
            if int_cbar:
                color_levels = np.arange(vmin-0.5, vmax+1)
                cmap = plt.cm.get_cmap(self.c_map, len(color_levels) - 1)
                norm = mpl.colors.BoundaryNorm(color_levels, cmap.N)

                im = ax.imshow(img, cmap=cmap, norm=norm, extent=extent, interpolation='none')
            else:
                im = ax.imshow(img, cmap=self.c_map, vmin=vmin, vmax=vmax, extent=extent, interpolation='none')

            if not hide_cbar:
                div = make_axes_locatable(ax)
                location = "right"
                pad = 0.1
                if cbar_orientation == "horizontal": 
                    location = "bottom"
                    if not hide_axes: 
                        pad = 0.6
                cax = div.append_axes(location, size=0.15, pad=pad)

                if log_scale:
                    cbar = plt.colorbar(
                        plt.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=self.c_map),
                        cax=cax,
                        orientation=cbar_orientation
                    )
                    plt.minorticks_off()
                else:
                    cbar = plt.colorbar(im, cax=cax, orientation=cbar_orientation)

                if int_cbar:
                    tick_positions = np.int32([tick + 0.5 for tick in color_levels[:-1]])
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels(tick_positions, fontsize=self.font_size)
                    cbar.ax.xaxis.set_ticks_position('none')
                    cbar.ax.yaxis.set_ticks_position('none')

                # cbar.set_ticks(np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])*1e8)
                # cbar.set_ticks(np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])*1e8)
                # cbar.set_ticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*1e8)
                cbar.ax.tick_params(labelsize=self.font_size)
                cbar.set_label(self.to_label_name(scalar_name), size=self.font_size)
                cbar.ax.tick_params(direction='in')
                # cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

            fig_name = self.dir_out + 'f_' + case_name + '_' + scalar_name + '_' + frame + '.' + savetype
        
        if hide_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("$x$ [m]", size=self.font_size)
            ax.set_ylabel("$y$ [m]", size=self.font_size)
        
        if self.zoomedin_view:
            axins = ax.inset_axes(
                [self.zoomedin_pos[0], self.zoomedin_pos[1], self.zoomedin_size[0], self.zoomedin_size[1]], 
                xlim=(self.zoomedin_box[0, 0], self.zoomedin_box[1, 0]), 
                ylim=(self.zoomedin_box[0, 1], self.zoomedin_box[1, 1])
            )
            axins.imshow(img_zoomedin, cmap=self.c_map, vmin=vmin, vmax=vmax, extent=self.zoomedin_bound, interpolation='none')
            self.format_ax(axins, hide_spines=hide_spines)
            axins.set_xticks([])
            axins.set_yticks([])
            if self.zoomedin_indication_lines:
                indicate_inset_zoom_obj = ax.indicate_inset_zoom(axins, edgecolor="red", linewidth=1.2, linestyle='--')
                for line in indicate_inset_zoom_obj[1]:
                    line.set_color('red')
                    line.set_linewidth(1.2)
                    line.set_linestyle('--')

        fig.tight_layout()
        if showfig: plt.show()
        if savefig: 
            fig.savefig(fig_name, dpi=300, bbox_inches="tight", transparent=True)
            plt.close()
            
        return fig_name

    def plotCp(self, file, savefig=False, plotref=False):
        mesh = pv.read(file)

        x       = np.array(mesh.cell_centers().points[:, 0])
        y       = np.array(mesh.cell_centers().points[:, 1])
        Cp      = np.array(mesh.cell_data["Aerodynamic/Cp"])
        ObjSurf = np.array(mesh.cell_data["Aerodynamic/ObjectSurface"]).astype(bool)

        x  = x [ObjSurf]
        y  = y [ObjSurf]
        Cp = Cp[ObjSurf]
        x  = (x - np.min(x))/(np.max(x) - np.min(x))    # normalize x

        fig, ax = plt.subplots()
        self.format_ax(ax)
        ax.invert_yaxis()

        ax.plot(x[y>0], Cp[y>0], 'o', c='#0987db', markersize=self.marker_size, markerfacecolor='none', label='upper')
        ax.plot(x[y<0], Cp[y<0], 'o', c='#cb3712', markersize=self.marker_size, markerfacecolor='none', label='lower')
        if plotref:
            x_ref  = pd.read_csv(self.dir_prj + '/vis/naca0012_Ma=0.8_ref.csv')['x']
            cp_ref = pd.read_csv(self.dir_prj + '/vis/naca0012_Ma=0.8_ref.csv')[' y']
            ax.plot(x_ref, cp_ref, 'k--', linewidth=self.line_width, label='reference')
        ax.set_xlabel(r'$\hat{x}$', fontsize=self.font_size)
        ax.set_ylabel(r'$C_p$', fontsize=self.font_size)
        ax.legend(loc='best', frameon=False, fontsize=self.font_size)
        if savefig:
            fig_name  = self.dir_out + self.__file_to_case_name__(file) + '_Cp.pdf'
            fig.savefig(fig_name, bbox_inches="tight", transparent=True)
        plt.show()

    def format_ax(self, ax, hide_spines=False): 
        if hide_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.spines['top'].set_linewidth(self.bound_width)
            ax.spines['bottom'].set_linewidth(self.bound_width)
            ax.spines['left'].set_linewidth(self.bound_width)
            ax.spines['right'].set_linewidth(self.bound_width)

        ax.axes.xaxis.set_ticks_position('both')
        ax.axes.yaxis.set_ticks_position('both')
        ax.tick_params(axis='x', direction='in', labelsize=self.font_size)
        ax.tick_params(axis='y', direction='in', labelsize=self.font_size)

    def latest_file(self, type='vtu'):
        list_of_files = glob.glob(self.dir_res + '*.' + type)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    
    def find_file(self, case_name, frame, type='vtu'):
        return self.dir_res + case_name + '_' + frame + '.' + type

    def load_observables(self, case_name):
        return pd.read_csv(self.dir_res + f'{case_name}_observables.csv')

    def __filter_edge__(self, mesh):
        edges = mesh.extract_all_edges()
        edge_points = edges.points

        # Get the connectivity (cells) of the edges
        cells = edges.lines.reshape((-1, 3))[:, 1:]

        # Filter edges that are aligned with the x or y axis
        aligned_edges = []
        for cell in cells:
            p1 = edge_points[cell[0]]
            p2 = edge_points[cell[1]]
            
            # Check if edge is aligned with the x-axis (y and z are the same for both points)
            if p1[1] == p2[1] and p1[2] == p2[2]:
                aligned_edges.append(cell)
            # Check if edge is aligned with the y-axis (x and z are the same for both points)
            elif p1[0] == p2[0] and p1[2] == p2[2]:
                aligned_edges.append(cell)
            # Check if edge is aligned with the z-axis (x and y are the same for both points)
            elif p1[0] == p2[0] and p1[1] == p2[1]:
                aligned_edges.append(cell)

        # Create a new PyVista PolyData object for the filtered edges
        filtered_edges = pv.PolyData()
        filtered_edges.points = edge_points
        filtered_edges.lines = np.hstack([[2] + list(edge) for edge in aligned_edges])

        return filtered_edges

    def __adjust_camera__(self, plotter, view):
        if view == '2d':
            plotter.camera.ParallelProjectionOn()
            plotter.camera_position = 'xy'
            plotter.camera.tight()

    def __find_closest_vertex__(self, mesh, p:np.ndarray):
        vertices = mesh.points

        dist_min = 1e5
        vertex_closest = None
        for vertex in vertices:
            v = np.array(vertex)
            dist = np.linalg.norm(v - p)
            if dist <= dist_min : 
                dist_min = dist
                vertex_closest = vertex

        return vertex_closest

    def __file_to_case_name__(self, file):
        return file.split('/')[-1].split('_')[0]

    def __file_to_frame__(self, file):
        return file.split('/')[-1].split('_')[-1].replace('.vtu', '')

#---------------------------------------------
def main():
    #####
    # result to plot
    #####
    # 1. plot latest result or
    # file = plab.latest_file(type='vtu')

    # 2. plot a specific result
    # case_name = 'naca'; frame= '0065'
    # case_name = 'cylinder'; frame= '0000'
    # case_name = 'cylinderg'; frame= '0047'
    # case_name = 'plate'; frame= '0001'
    # case_name = 'diamond'; frame= '0001'
    # case_name = 'scramjet'; frame= '0013'
    # case_name = 'scramjetviscous'; frame= '0020'
    # case_name = "shockBubble"; frame = '0024'
    # case_name = "shockDrop3d"; frame = "0035"
    case_name = "oscDrop"; frame = "0197"
    # case_name = "multi-shock-tube"; frame = "0001"

    plab = plotlab(model="multi2d", 
                   case_name=case_name,
                c_map=colormaps.parula, 
            #    c_map="magma",
            #    c_map="Blues",
            #    c_map="turbo",
                edge_color="gray")

    file = plab.find_file(case_name, frame, type='vtu')
    
    #####
    # generate plot
    #####
    # zoomedin_box = np.array([[-0.55, -0.14, -0.005], [0.55, 0.14, 0.005]])
    # zoomedin_box = np.array([[-0.56, -0.06, -0.005], [-0.44, 0.06, 0.005]])

    # [0.025, -0.33], [0.95, 0.95] for naca

    # plab.add_zoomedin_view(zoomedin_box, [0.02, 0.02], [0.4, 0.4])
    # plab.plot2d(file, scalar_name="p_3", showgrid=False, savefig=True, hide_axes=True
    #             ,v_min=0.0, v_max=4.0)
                # ,zoomedin_box=np.array([[-1.0, -1.0, -0.005], [1.0, 1.0, 0.005]]))

    # grid only
    # plab.add_zoomedin_view(np.array([[-0.55, 0.04, -0.005], [-0.43, 0.16, 0.005]]), 
    #                        [0.02, 0.02], [0.4, 0.4],
    #                        indication_lines=True) # plate
    # plab.add_zoomedin_view(np.array([[-2.25, -0.25, -0.005], [-1.75, 0.25, 0.005]]), 
    #                        [0.55, 0.55], [0.45, 0.45],
    #                        indication_lines=False) # line
    # plab.add_zoomedin_view(np.array([[2.925, -0.3125, -0.005], [3.025, -0.2875, 0.005]]), 
    #                        [0.8, 0.75], [0.275, 0.275],
    #                        indication_lines=True) # scramjet
    # plab.plot2d(file, showgrid=True, savefig=True, hide_axes=True, showfig=False, hide_spines=False) 


    # plab.add_zoomedin_view(np.array([[2.925, -0.3125, -0.005], [3.025, -0.2875, 0.005]]), 
    #                        [0.725, 0.725], [0.275, 0.275],
    #                        indication_lines=True) # scramjet
    # plab.plot2d(file, 
    #             scalar_name="acti_level", 
    #             v_min=0, v_max=9,
    #             plottype="half_grid",
    #             cbar_orientation="horizontal", 
    #             int_cbar=True,
    #             # log_scale=True,
    #             showgrid=False, showfig=True, savefig=True, hide_axes=True, hide_cbar=False, hide_spines=False,
    #             zoomedin_box=np.array([[-1.5, -0.5, -0.005], [-0.5, 0.5, 0.005]]))


    # plab.plotCp(file, plotref=False, savefig=True)

    # plab.animation(case_name, "p_1", "2d", 
    #             #    plottype="schlieren",
    #                v_max=140, v_min=-20, 
    #                framerate=30, skip_plotting=True, create_videos=True,
    #                keepfigs=True, hideaxes=True, hidecbar=True, showgrid=False,
    #                zoomedin_box=np.array([[-3.0, -1.5, -0.005], [4.0, 1.5, 0.005]]))


    # plab.plot1d(file, case_name, "acti_level", plotref=True, plotinit=False, file_ref="shock_tube.csv", img_top=False, show_legend=False)

    plab.plotObservables(case_name, "ke", startframe=0)

    # plab.animation(case_name, "ke", "1d", framerate=60, skip_plotting=True, keepfigs=True)


    # plab.pvplot3d(file, "p_5", v_max=5e5, v_min=1e5, plottype="contour")
    # plab.pvplot3d(file, scalar_name="p_6", 
    #               v_max=1, v_min=0,
    #               showgrid=True, savefig=True,
    #               clipbox=np.array([[-2.0, -1.0, 0.0], [2.0, 0.0, 1.0]]))

    # plab.animation("shockDrop3d", "p_6", '3d', v_min=0.0, v_max=1.0)

    # plab.histogram(file, case_name, bins=9, showfig=True, savefig=True)


if __name__ == "__main__":
    main()