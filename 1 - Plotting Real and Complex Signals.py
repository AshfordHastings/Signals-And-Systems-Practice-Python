import numpy as np
import matplotlib.pyplot as plt

# Plotting a continuous signal of x(t)=325*sin(2*pi*50t)

t = np.linspace(-0.02, 0.05, 1000)

plt.plot(t,325*np.sin(2*np.pi*50*t))

plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(r'Plot of CT signal $x(t)=325 \sin(2\pi 50 t)$');
plt.xlim([-0.02, 0.05]);
plt.show()

# Plotting a discrete signal of x(t)=325*sin(2*pi*50t)

n = np.arange(50);
dt = 0.07/50
x = np.sin(2 * np.pi * 50 * n * dt)
plt.xlabel('n');
plt.ylabel('x[n]');
plt.title(r'Plot of DT signal $x[n] = 325 \sin(2\pi 50 n \Delta t)$');
plt.stem(n, x);

# Plotting complex signal x(t)=e^(j100*pi*t)

t = np.linspace(-.02, 0.05, 100)
plt.subplot(2,1,1); plt.plot(t, np.exp(2j*np.pi*50*t).real );
plt.xlabel('t');
plt.ylabel('Re x(t)');
plt.title(r'Real part of $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.subplot(2,1,2); plt.plot(t, np.exp(2j*np.pi*50*t).imag );
plt.xlabel('t');
plt.ylabel('Imag x(t)');
plt.title(r'Imaginary part of $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.show();

# Plotting real and imaginary components of complex signal

t = np.linspace(-.02, 0.05, 100)
plt.subplot(2,1,1); plt.plot(t, np.abs(np.exp(2j*np.pi*50*t)) );
plt.xlabel('t');
plt.ylabel('Magnitude x(t)');
plt.title('Magnitude of $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.subplot(2,1,2); plt.plot(t, np.angle(np.exp(2j*np.pi*50*t))*360/(2*np.pi));
plt.xlabel('t');
plt.ylabel(r'$\angle x(t)$');
plt.title('Phase of $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);

plt.tight_layout();
plt.show();

# Plot the pulse function 3pulse(t-2)
n = np.arange(10);
x = np.zeros_like(n);
x[2] = 3;
#plt.vlines(n,0,x,'b');
plt.ylim(-1,4);
plt.plot(n, x, 'b');
plt.show();