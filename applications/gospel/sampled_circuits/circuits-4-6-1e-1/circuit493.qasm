OPENQASM 2.0;
include "qelib1.inc";
qreg q494[4];
rx(3*pi/2) q494[3];
cx q494[3],q494[2];
cx q494[2],q494[1];
cx q494[0],q494[1];
rx(pi/4) q494[1];
