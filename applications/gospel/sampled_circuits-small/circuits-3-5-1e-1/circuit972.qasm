OPENQASM 2.0;
include "qelib1.inc";
qreg q973[3];
rx(7*pi/4) q973[2];
cx q973[1],q973[2];
cx q973[1],q973[0];
