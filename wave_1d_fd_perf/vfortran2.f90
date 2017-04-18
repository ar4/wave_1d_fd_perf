module vfortran2

  implicit none
  !private
  !public :: step

contains

  subroutine step(f1, f2, model_padded, dt, dx, sources, sources_x,    &
      num_steps, nx_padded, num_sources, source_len)

    integer, intent (in) :: nx_padded
    integer, intent (in) :: num_sources
    integer, intent (in) :: source_len
    real, intent (in out), dimension (nx_padded) :: f1
    real, intent (in out), dimension (nx_padded) :: f2
    real, intent (in), dimension (nx_padded) :: model_padded
    real, intent (in) :: dt
    real, intent (in) :: dx
    real, intent (in), dimension (num_sources, source_len) :: sources
    integer, intent (in), dimension (num_sources) :: sources_x
    integer, intent (in) :: num_steps

    integer :: step_idx
    logical :: even

    do step_idx = 1, num_steps
    even = (mod (step_idx, 2) == 0)
    if (even) then
      call step_inner(f2, f1, model_padded, dt, dx, sources, sources_x,&
        step_idx, nx_padded, num_sources, source_len)
    else
      call step_inner(f1, f2, model_padded, dt, dx, sources, sources_x,&
        step_idx, nx_padded, num_sources, source_len)
    end if
    end do

  end subroutine step

  subroutine step_inner(f, fp, model_padded, dt, dx, sources,          &
      sources_x, step_idx, nx_padded, num_sources, source_len)

    integer, intent (in) :: nx_padded
    integer, intent (in) :: num_sources
    integer, intent (in) :: source_len
    real, intent (in out), dimension (nx_padded) :: f
    real, intent (in out), dimension (nx_padded) :: fp
    real, intent (in), dimension (nx_padded) :: model_padded
    real, intent (in) :: dt
    real, intent (in) :: dx
    real, intent (in), dimension (num_sources, source_len)  :: sources
    integer, intent (in), dimension (num_sources) :: sources_x
    integer, intent (in) :: step_idx

    integer :: i
    integer :: sx
    real :: f_xx

    !$omp parallel do private(f_xx)
    do i = 9, nx_padded - 8
    f_xx = (                                                           &
      -735*f(i-8)+15360*f(i-7)                                         &
      -156800*f(i-6)+1053696*f(i-5)                                    & 
      -5350800*f(i-4)+22830080*f(i-3)                                  & 
      -94174080*f(i-2)+538137600*f(i-1)                                & 
      -924708642*f(i+0)                                                & 
      +538137600*f(i+1)-94174080*f(i+2)                                & 
      +22830080*f(i+3)-5350800*f(i+4)                                  & 
      +1053696*f(i+5)-156800*f(i+6)                                    & 
      +15360*f(i+7)-735*f(i+8))/(302702400*dx**2)
    fp(i) = (model_padded(i)**2 * dt**2 * f_xx + 2 * f(i) - fp(i))
    end do

    do i = 1, num_sources
    sx = sources_x(i) + 9;
    fp(sx) = fp(sx) + (model_padded(sx)**2 * dt**2                     &
      * sources(i, step_idx))
    end do
  end subroutine step_inner

end module vfortran2
